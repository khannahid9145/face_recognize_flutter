// lib/face_first_page.dart
import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';

import 'package:face/screens/verify_face.dart';
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

class FaceRegister extends StatefulWidget {
  const FaceRegister({super.key});

  @override
  State<FaceRegister> createState() => _FaceRegisterState();
}

class _FaceRegisterState extends State<FaceRegister> {
  CameraController? _cam;
  bool _busy = false;
  File? _lastImage;
  List<Face> _faces = [];
  int _imgW = 0, _imgH = 0;
  FaceEmbedder? _embedder;
  Float32List? _embedding;
  Uint8List? _uprightBytes;

  final FaceDetector _detector = FaceDetector(
    options: FaceDetectorOptions(
      enableLandmarks: true,
      enableContours: true,
      enableClassification: true,
    ),
  );

  @override
  void initState() {
    super.initState();
    if (kDebugMode) {
      print('FaceRegister: initState');
    }
    _initialize();
  }

  Float32List _toFloat32List(dynamic v) {
    final list = (v as List).map((e) => (e as num).toDouble()).toList();
    return Float32List.fromList(list);
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

  Future<void> _initialize() async {
    _embedder ??= FaceEmbedder(
      inputShape: const [1, 112, 112, 3],
      embeddingSize: 128,
    );

    if (kDebugMode) {
      print('FaceRegister: _initialize (model + camera)');
    }

    try {
      await _embedder!.load();
      if (kDebugMode) {
        print('FaceRegister: model loaded successfully');
      }
      await Future.delayed(const Duration(milliseconds: 150));
      await _initCamera();
    } catch (e) {
      if (!mounted) return;
      if (kDebugMode) {
        print('FaceRegister: Initialization error: $e');
      }
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Initialization error: $e')));
    }
  }

  Future<void> _cleanUp() async {
    try {
      if (kDebugMode) {
        print('FaceRegister: _cleanUp');
      }
      if (_cam != null) {
        await _cam!.dispose();
        _cam = null;
        if (kDebugMode) {
          print('FaceRegister: Camera disposed');
        }
      }

      if (_embedder != null) {
        _embedder!.close();
        _embedder = null;
        if (kDebugMode) {
          print('FaceRegister: Embedder disposed');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        print('FaceRegister: Cleanup error: $e');
      }
    }
  }

  Future<void> _initCamera() async {
    if (_cam != null) {
      await _cam!.dispose();
      _cam = null;
    }

    try {
      if (kDebugMode) {
        print('FaceRegister: Initializing camera...');
      }
      final cams = await availableCameras();
      if (kDebugMode) {
        print('FaceRegister: Available cameras: ${cams.length}');
      }
      if (cams.isEmpty) {
        throw CameraException('No cameras', 'No cameras available on device');
      }

      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      if (kDebugMode) {
        print(
          'FaceRegister: Selected camera: ${front.name} (${front.lensDirection})',
        );
      }

      final ctrl = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      try {
        await ctrl.initialize();
      } on CameraException catch (e) {
        // Common case: camera still "in use" right after coming back
        if (kDebugMode) {
          print('FaceRegister: CameraException on initialize: ${e.code} / $e');
        }
        // Simple retry after a short delay
        if (e.code == 'CameraAccess' || e.code == 'CameraInUse') {
          await Future.delayed(const Duration(milliseconds: 300));
          if (!mounted) return;
          if (kDebugMode) {
            print('FaceRegister: retrying camera initialize...');
          }
          await ctrl.initialize();
        } else {
          rethrow;
        }
      }

      if (!mounted) return;

      try {
        await ctrl.lockCaptureOrientation();
        if (kDebugMode) {
          print('FaceRegister: Camera initialized and locked');
        }
      } catch (e) {
        if (kDebugMode) {
          print('FaceRegister: Warning: Could not lock orientation: $e');
        }
      }

      if (!mounted) return;
      setState(() {
        _cam = ctrl;
        _busy = false;
      });
    } catch (e) {
      if (!mounted) return;
      if (kDebugMode) {
        print('FaceRegister: _initCamera error: $e');
      }
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Camera error: $e')));
    }
  }

  Future<void> _saveEnrollment(Float32List emb) async {
    await _secure.write(key: _userKey, value: jsonEncode(emb));
  }

  Future<void> _saveImage() async {
    if (_embedding == null) return;

    final endpoint = Uri.parse(
      'https://developer.tickleright.in/app_routes/employeeRoute.php?action=update_face_embbed',
    );
    final embList = _embedding!.toList();
    final payload = {"contact_id": "3183", "embedding": embList};

    try {
      final resp = await http
          .post(
            endpoint,
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 20));

      if (resp.statusCode == 200) {
        var decodedBody = jsonDecode(resp.body); 
        if (decodedBody['error'] == 0) {
            _login();
        } else {
          print('error is 1, error in daving the embedding');
        }
        if (kDebugMode) {
          print('✅ Embedding saved: ${resp.body}');
        }
      } else {
        if (kDebugMode) {
          print('❌ Server error ${resp.statusCode}: ${resp.body}');
        }
        throw Exception('Server ${resp.statusCode}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('❌ Network error: $e');
      }
      rethrow;
    }
  }

  void _clearState() {
    if (!mounted) return;
    setState(() {
      _lastImage = null;
      _faces = [];
      _embedding = null;
      _uprightBytes = null;
      _imgW = 0;
      _imgH = 0;
    });
  }

  Future<void> _re_capture() async {
    if (kDebugMode) {
      print('FaceRegister: re-capture');
    }
    if (!mounted) return;
    setState(() {
      _busy = true;
    });

    _clearState();
    await _cleanUp();

    if (mounted) {
      await _initialize();
    }

    if (mounted) {
      setState(() => _busy = false);
    }
  }

  Future<void> _login() async {
    if (kDebugMode) {
      print('FaceRegister: login btn clicked');
    }

    // 1️⃣ clean camera + interpreter before navigation
    await _cleanUp();

    if (!mounted) return;

    await Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const VerifyFace()),
    );

    // 2️⃣ Coming back: give OS a moment to fully release camera,
    // then reset state and re-init.
    if (!mounted) return;

    await Future.delayed(const Duration(milliseconds: 300));
    _clearState();
    await _initialize();
  }

  Future<void> _captureAndDetect() async {
    if (_cam == null || _busy) return;
    if (!mounted) return;

    setState(() => _busy = true);

    try {
      if (!_cam!.value.isInitialized) {
        throw CameraException('Not initialized', 'Camera is not initialized');
      }

      final file = await _cam!.takePicture();
      final f = File(file.path);
      if (!f.existsSync()) {
        throw Exception('Image file not found: ${file.path}');
      }

      final input = InputImage.fromFile(f);
      final faces = await _detector.processImage(input);

      if (faces.isEmpty) {
        if (!mounted) return;
        setState(() {
          _lastImage = f;
          _faces = const [];
          _uprightBytes = null;
        });
        return;
      }

      final bytes = await f.readAsBytes();
      final decodedRaw = img.decodeImage(bytes);
      if (decodedRaw == null) throw Exception('Failed to decode image');
      final upright = img.bakeOrientation(decodedRaw);

      _imgW = upright.width;
      _imgH = upright.height;
      _uprightBytes = Uint8List.fromList(img.encodeJpg(upright, quality: 95));

      final target = faces.reduce(
        (a, b) =>
            a.boundingBox.width * a.boundingBox.height >
                    b.boundingBox.width * b.boundingBox.height
                ? a
                : b,
      );

      if (_embedder == null) {
        throw Exception('Embedder not initialized');
      }

      final face112 = _cropToModelInput(upright, target.boundingBox);
      final emb = _embedder!.run(face112);
      await _saveEnrollment(emb);

      if (!mounted) return;
      setState(() {
        _embedding = emb;
        _lastImage = f;
        _faces = faces;
      });
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
      print('FaceRegister: dispose');
    }

    _detector.close();
    _cam?.dispose();
    _cam = null;
    _embedder?.close();
    _embedder = null;

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _cam;

    if (cam == null || !cam.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Register Face')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                if (_lastImage != null) ...[
                  ElevatedButton.icon(
                    onPressed: _busy ? null : _re_capture,
                    icon: const Icon(Icons.camera_alt),
                    label: Text(_busy ? 'Working…' : 'Re-Capture'),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton.icon(
                    onPressed: _busy ? null : _saveImage,
                    icon: const Icon(Icons.save),
                    label: Text(_busy ? 'Saving…' : 'Register Face'),
                  ),
                ] else ...[
                  ElevatedButton.icon(
                    onPressed: _busy ? null : _captureAndDetect,
                    icon: const Icon(Icons.camera_alt),
                    label: Text(_busy ? 'Working…' : 'Capture'),
                  ),
                ],
                const SizedBox(width: 8),
                ElevatedButton.icon(
                  onPressed: _busy ? null : _login,
                  icon: const Icon(Icons.login),
                  label: Text(_busy ? 'Processing…' : 'Login'),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),
          if (_lastImage == null)
            Expanded(
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: cam.value.previewSize!.height,
                  height: cam.value.previewSize!.width,
                  child: CameraPreview(cam),
                ),
              ),
            ),
          const SizedBox(height: 8),
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
                          padding: const EdgeInsets.symmetric(horizontal: 16.0),
                          child: Align(
                            alignment: Alignment.centerLeft,
                            child: Text(
                              'Detected faces: ${_faces.length}',
                              style: Theme.of(context).textTheme.titleMedium,
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
    );
  }
}
