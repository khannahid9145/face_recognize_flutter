// lib/face_first_page.dart
import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as math;

import '../ml/face_embedder.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

final _secure = const FlutterSecureStorage();
const _userKey = 'face_template_user1';

img.Image _cropToModelInput(img.Image full, Rect bb, {int size = 112}) {
  final x = bb.left.floor().clamp(0, math.max(0, full.width - 1));
  final y = bb.top.floor().clamp(0, math.max(0, full.height - 1));
  final w = bb.width.ceil().clamp(1, full.width - x);
  final h = bb.height.ceil().clamp(1, full.height - y);

  final face = img.copyCrop(
    full,
    x: x.toInt(),
    y: y.toInt(),
    width: w.toInt(),
    height: h.toInt(),
  );
  return img.copyResize(face, width: size, height: size);
}

class VerifyFace extends StatefulWidget {
  const VerifyFace({super.key});

  @override
  State<VerifyFace> createState() => _VerifyFaceState();
}

class _VerifyFaceState extends State<VerifyFace> {
  CameraController? _cameraController;
  bool _busy = false;
  File? _lastImage;
  bool _faceAuthorize = false;
  List<Face> _faces = [];
  int _imgW = 0, _imgH = 0;
  FaceEmbedder? _embedder;
  Float32List? _embedding;
  Float32List? _storedEmbedding;
  Uint8List? _uprightBytes;

  final FaceDetector _detector = FaceDetector(
    options: FaceDetectorOptions(
      enableLandmarks: true,
      enableContours: true,
      enableClassification: true,
    ),
  );

  Float32List _toFloat32List(dynamic v) {
    final list = (v as List).map((e) => (e as num).toDouble()).toList();
    return Float32List.fromList(list);
  }

  @override
  void initState() {
    super.initState();
    if (kDebugMode) {
      print('Initializing VerifyFace...');
    }
    _initialize();
    getEmbedValue();
  }

  Future<void> getEmbedValue() async {
    final endpoint = Uri.parse(
      'https://developer.tickleright.in/app_routes/employeeRoute.php?action=get_embedding_value',
    );
    final payload = {"contact_id": "3183"};
    try {
      final response = await http
          .post(
            endpoint,
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 20));

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        final raw = decoded['data']['embedding'];
        final embeddingList = raw is String ? jsonDecode(raw) : raw;
        if (!mounted) return; // <- important when page is popped
        setState(() {
          _storedEmbedding = _toFloat32List(embeddingList);
        });
      } else {
        if (kDebugMode) {
          print('❌ Server error ${response.statusCode}: ${response.body}');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('❌ Network error: $e');
      }
    }
  }

  double cosineSimilarityInline(Float32List a, Float32List b) {
    assert(a.length == b.length);
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    final denom = math.sqrt(na) * math.sqrt(nb);
    return denom == 0 ? 0.0 : dot / denom;
  }

  Future<void> verifyFaceEmbed(Float32List currentEmb) async {
    if (_storedEmbedding == null) {
      await getEmbedValue(); // loads into _storedEmbedding
    }
    if (_storedEmbedding == null) {
      throw Exception('No stored embedding found.');
    }

    final sim = cosineSimilarityInline(currentEmb, _storedEmbedding!);

    if (sim > 0.8) {
      if (!mounted) return;
      setState(() {
        _faceAuthorize = true;
      });
      if (kDebugMode) {
        print('✅ Match (similarity=$sim)');
      }
    } else {
      if (kDebugMode) {
        print('❌ Not a match (similarity=$sim)');
      }
    }
  }

  Future<void> _initialize() async {
    _embedder = FaceEmbedder(
      inputShape: const [1, 112, 112, 3],
      embeddingSize: 128,
    );

    try {
      await _embedder!.load(); // wait for model to load
      if (kDebugMode) {
        print('Model loaded successfully');
      }
      await _initCamera(); // then start camera
    } catch (e) {
      if (!mounted) return;
      if (kDebugMode) {
        print('Initialization error: $e');
      }
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Initialization error: $e')));
    }
  }

  Future<void> _initCamera() async {
    if (_cameraController != null) {
      await _cameraController!.dispose();
    }

    try {
      if (kDebugMode) {
        print('Initializing camera...');
      }
      final cams = await availableCameras();
      if (kDebugMode) {
        print('Available cameras: ${cams.length}');
      }
      if (cams.isEmpty) {
        throw CameraException('No cameras', 'No cameras available on device');
      }

      // Prefer front camera for auth; fallback to first.
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      if (kDebugMode) {
        print('Selected camera: ${front.name} (${front.lensDirection})');
      }

      final ctrl = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      // Wait for controller to initialize
      try {
        await ctrl.initialize();
      } catch (e) {
        if (kDebugMode) {
          print('Error initializing camera controller: $e');
        }
        throw CameraException('Init failed', 'Could not initialize camera: $e');
      }

      if (!mounted) return;

      // Ensure the camera is properly locked for capture
      try {
        await ctrl.lockCaptureOrientation();
        if (kDebugMode) {
          print('Camera initialized and locked');
        }
      } catch (e) {
        if (kDebugMode) {
          print('Warning: Could not lock camera orientation: $e');
        }
      }

      if (!mounted) return;
      setState(() {
        _cameraController = ctrl;
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Camera error: $e')));
    }
  }

  Future<void> _saveEnrollment(Float32List emb) async {
    await _secure.write(key: _userKey, value: jsonEncode(emb));
  }

  void _clearState() {
    if (!mounted) return;
    setState(() {
      _lastImage = null;
      _faces = [];
      _embedding = null;
      _storedEmbedding = null; // clear in-memory embedding
      _uprightBytes = null;
      _faceAuthorize = false;
      _imgW = 0;
      _imgH = 0;
    });
  }

  Future<void> _reCapture() async {
    _clearState();
  }

  Future<void> _captureAndDetect() async {
    if (_cameraController == null || _busy) return;
    setState(() => _busy = true);

    try {
      if (!_cameraController!.value.isInitialized) {
        throw CameraException('Not initialized', 'Camera is not initialized');
      }

      final file = await _cameraController!.takePicture();
      final f = File(file.path);
      if (!f.existsSync()) {
        throw Exception('Image file not found: ${file.path}');
      }

      // 1) Detect faces from file (ML Kit reads EXIF and returns upright-space boxes)
      final input = InputImage.fromFile(f);
      final faces = await _detector.processImage(input);
      if (faces.isEmpty) {
        if (!mounted) return;
        setState(() {
          _lastImage = null;
          _faces = const [];
          _uprightBytes = null;
        });
        return;
      }

      // 2) Decode and BAKE orientation to get upright pixels for math & display
      final bytes = await f.readAsBytes();
      final decodedRaw = img.decodeImage(bytes);
      if (decodedRaw == null) throw Exception('Failed to decode image');
      final upright = img.bakeOrientation(decodedRaw);

      // Keep these for scaling + showing
      _imgW = upright.width;
      _imgH = upright.height;
      _uprightBytes = Uint8List.fromList(img.encodeJpg(upright, quality: 95));

      // 3) Pick largest face and crop FROM THE UPRIGHT IMAGE
      final target = faces.reduce(
        (a, b) =>
            a.boundingBox.width * a.boundingBox.height >
                    b.boundingBox.width * b.boundingBox.height
                ? a
                : b,
      );

      final face112 = _cropToModelInput(upright, target.boundingBox);

      // 4) Embed and store
      if (_embedder == null) {
        throw Exception('Embedder not initialized');
      }
      final emb = _embedder!.run(face112);
      await _saveEnrollment(emb);

      if (!mounted) return;
      setState(() {
        _embedding = emb;
        _lastImage = f;
        _faces = faces; // boxes are in upright-space
      });
      

      await verifyFaceEmbed(emb);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Detect error: $e')));
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  void dispose() {
    if (kDebugMode) {
      print('VerifyFace dispose');
    }

    // clear UI state (no setState here – widget is being disposed)
    _lastImage = null;
    _faces = [];
    _embedding = null;
    _storedEmbedding = null;
    _uprightBytes = null;
    _faceAuthorize = false;

    _detector.close();
    _cameraController?.dispose();
    _embedder?.close();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _cameraController;

    return WillPopScope(
      onWillPop: () async {
        // clear everything when user goes back
        _clearState();
        return true;
      },
      child: Scaffold(
        appBar: AppBar(title: const Text('Authorize Face')),
        body: Column(
          children: [
            // 1) Capture / Recapture buttons
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16.0),
              child: Row(
                children: [
                  if (_faceAuthorize == false && _lastImage != null)
                    ElevatedButton.icon(
                      onPressed: _busy ? null : _reCapture,
                      icon: const Icon(Icons.camera_alt),
                      label: Text(_busy ? 'Working…' : 'Re-Capture'),
                    )
                  else if (_lastImage == null)
                    ElevatedButton.icon(
                      onPressed: _busy ? null : _captureAndDetect,
                      icon: const Icon(Icons.camera_alt),
                      label: Text(_busy ? 'Working…' : 'Capture'),
                    ),
                  const SizedBox(width: 8),
                ],
              ),
            ),

            const SizedBox(height: 8),

            // 2) Live camera preview (before capture)
            if (_lastImage == null)
              Expanded(
                child: cam == null || !cam.value.isInitialized
                    ? const Center(child: CircularProgressIndicator())
                    : FittedBox(
                        fit: BoxFit.cover,
                        child: SizedBox(
                          width: cam.value.previewSize!.height,
                          // note swapped width/height
                          height: cam.value.previewSize!.width,
                          child: CameraPreview(cam),
                        ),
                      ),
              ),

            const SizedBox(height: 8),

            // 3) Captured image and overlays (ONLY after capture)
            if (_lastImage != null)
              Expanded(
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final displayW = constraints.maxWidth;
                    final displayH =
                        displayW * (_imgH == 0 ? 4 / 3 : _imgH / _imgW);
                    final sx = _imgW == 0 ? 1.0 : displayW / _imgW;
                    final sy = _imgH == 0 ? 1.0 : displayH / _imgH;

                    return SingleChildScrollView(
                      child: Column(
                        children: [
                          SizedBox(
                            width: displayW,
                            height: displayH,
                            child: Stack(
                              children: [
                                if (_uprightBytes != null)
                                  Image.memory(
                                    _uprightBytes!,
                                    width: displayW,
                                    height: displayH,
                                    fit: BoxFit.contain,
                                  )
                                else
                                  const SizedBox.shrink(),

                                // Overlays (same upright coordinate space)
                                for (final face in _faces)
                                  Positioned(
                                    left: face.boundingBox.left * sx,
                                    top: face.boundingBox.top * sy,
                                    child: Container(
                                      width: face.boundingBox.width * sx,
                                      height: face.boundingBox.height * sy,
                                      decoration: BoxDecoration(
                                        border: Border.all(width: 2),
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 8),
                          Padding(
                            padding:
                                const EdgeInsets.symmetric(horizontal: 16.0),
                            child: Align(
                              alignment: Alignment.center,
                              child: Text(
                                _faceAuthorize
                                    ? 'Face Authorized Successfully 🎉'
                                    : 'Face not Authorized 😌',
                                style:
                                    Theme.of(context).textTheme.titleMedium,
                              ),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}
