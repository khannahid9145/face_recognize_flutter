// lib/face_first_page.dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as math;
import '../ml/face_embedder.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'dart:convert';
final _secure = const FlutterSecureStorage();
const _userKey = 'face_template_user1';

img.Image _cropToModelInput(img.Image full, Rect bb, {int size = 112}) {
  final x = bb.left.floor().clamp(0, math.max(0, full.width - 1));
  final y = bb.top.floor().clamp(0, math.max(0, full.height - 1));
  final w = bb.width.ceil().clamp(1, full.width - x);
  final h = bb.height.ceil().clamp(1, full.height - y);

  final face = img.copyCrop(full, x: x.toInt(), y: y.toInt(), width: w.toInt(), height: h.toInt());
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
    late final FaceEmbedder _embedder;

  final FaceDetector _detector = FaceDetector(
    options: FaceDetectorOptions(
      enableLandmarks: true,
      enableContours: true,
      enableClassification: true,
      // performanceMode: FaceDetectorMode.accurate, // optional
      // minFaceSize: 0.15, // optional
    ),
  );

  @override
  void initState() {
    _embedder = FaceEmbedder(
      inputShape: const [1, 112, 112, 3], // adjust if your model differs
      embeddingSize: 128,                 // adjust if your model differs
    );

    debugPrint('Initializing file...');
    super.initState();
    _initCamera();
  }


  Future<void> saveImage() async {
    debugPrint('Image save requested');
    
  }
  Future<void> _initCamera() async {
    if (_cam != null) {
      await _cam!.dispose();
    }
    
    try {
      debugPrint('Initializing camera...');
      final cams = await availableCameras();
      debugPrint('Available cameras: ${cams.length}');
      if (cams.isEmpty) {
        throw CameraException('No cameras', 'No cameras available on device');
      }
      
      // Prefer front camera for auth; fallback to first.
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      debugPrint('Selected camera: ${front.name} (${front.lensDirection})');
      
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
        debugPrint('Error initializing camera controller: $e');
        throw CameraException('Init failed', 'Could not initialize camera: $e');
      }
      
      if (!mounted) return;
      
      // Ensure the camera is properly locked for capture
      try {
        await ctrl.lockCaptureOrientation();
        debugPrint('Camera initialized and locked');
      } catch (e) {
        debugPrint('Warning: Could not lock camera orientation: $e');
      }
      
      setState(() {
        _cam = ctrl;
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

  

  Future<void> _captureAndDetect() async {
    if (_cam == null || _busy) {
      debugPrint('Camera not ready or busy');
      return;
    }
    
    setState(() => _busy = true);
    try {
      // Ensure camera is initialized
      if (!_cam!.value.isInitialized) {
        throw CameraException('Not initialized', 'Camera is not initialized');
      }

      debugPrint('Taking picture...');
      final file = await _cam!.takePicture();
      debugPrint('Picture taken, processing...');
      
      final f = File(file.path);
      if (!f.existsSync()) {
        throw Exception('Image file not found: ${file.path}');
      }
      
      // decode to get width/height for overlay scaling
      final bytes = await f.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Unable to decode image');
      _imgW = decoded.width;
      _imgH = decoded.height;

      final input = InputImage.fromFile(f);
      final faces = await _detector.processImage(input);

      debugPrint('Detected ${faces.length} faces');
      if (faces.isNotEmpty) {
        debugPrint('Face bounding boxes: ');
        final target = faces.reduce((a, b) =>
        a.boundingBox.width * a.boundingBox.height >
        b.boundingBox.width * b.boundingBox.height ? a : b);

        // decode full JPEG once (you already did earlier as `decoded`)
        // Make sure the bounding box is in the same coordinate space (ML Kit from file is in image space).
        final face112 = _cropToModelInput(decoded, target.boundingBox);

        // Get the 128-D (or whatever your model outputs) embedding
        final emb = _embedder.run(face112); // Float32List
        debugPrint('Embedding: $emb');
        // Example: store it or compare it
        await _saveEnrollment(emb); // see below
        setState(() {
          _lastImage = f;
          _faces = faces;
        });
      } else {
        setState(() {
          _lastImage = f;
          _faces = [];
        });
      }
      for (final face in faces) {
        final bb = face.boundingBox;
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Detect error: $e')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  void dispose() {
    _detector.close();
    _cam?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _cam;
    return Scaffold(
      appBar: AppBar(title: const Text('Register Face')),
      body: Column(
        children: [
          // Camera preview
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: ElevatedButton.icon(
              onPressed: _busy ? null : _captureAndDetect,
              icon: const Icon(Icons.camera_alt),
              label: Text(_busy ? 'Working…' : 'Capture & Detect'),
            ),
          ),
          AspectRatio(
            aspectRatio:
                cam?.value.previewSize != null
                    ? cam!.value.previewSize!.width /
                        cam.value.previewSize!.height
                    : 3 / 4,
            child:
                cam == null || !cam.value.isInitialized
                    ? const Center(child: CircularProgressIndicator())
                    : CameraPreview(cam),
          ),
          const SizedBox(height: 8),

          // Capture button
          const SizedBox(height: 8),
          // Detected image with overlays (if any)
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
                        Stack(
                          children: [
                            Image.file(
                              _lastImage!,
                              width: displayW,
                              height: displayH,
                              fit: BoxFit.contain,
                            ),
                            // Bounding boxes
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
                        const SizedBox(height: 8),
                        // Coordinates list
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
                        ListView.separated(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          itemCount: _faces.length,
                          separatorBuilder: (_, __) => const Divider(height: 1),
                          itemBuilder: (context, i) {
                            final f = _faces[i];
                            final bb = f.boundingBox;
                            return ListTile(
                              title: Text('Face #${i + 1}'),
                              subtitle: Text(
                                'box: L=${bb.left.toStringAsFixed(1)}, '
                                'T=${bb.top.toStringAsFixed(1)}, '
                                'W=${bb.width.toStringAsFixed(1)}, '
                                'H=${bb.height.toStringAsFixed(1)}',
                              ),
                            );
                          },
                        ),
                        const SizedBox(height: 16),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 16.0),
                          child: TextButton(onPressed: saveImage, child: Text('Save Image'))
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
