// lib/face_first_page.dart
import 'dart:io';
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

enum _PoseStep { center, left, right, up }

String _poseInstructionForStep(_PoseStep step) {
  switch (step) {
    case _PoseStep.center:
      return 'Look straight ahead';
    case _PoseStep.left:
      return 'Turn your head left';
    case _PoseStep.right:
      return 'Turn your head right';
    case _PoseStep.up:
      return 'Tilt your head up slightly';
  }
}

class _FacePoseRingPainter extends CustomPainter {
  _FacePoseRingPainter(this.progress);

  final double progress;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = math.min(size.width, size.height) / 2 - 12;
    final tickPaint =
        Paint()
          ..strokeCap = StrokeCap.round
          ..strokeWidth = 3;
    const tickCount = 60;
    final activeTicks = (tickCount * progress.clamp(0, 1)).round();

    for (int i = 0; i < tickCount; i++) {
      final angle = (2 * math.pi * (i / tickCount)) - math.pi / 2;
      final start = Offset(
        center.dx + math.cos(angle) * (radius - 12),
        center.dy + math.sin(angle) * (radius - 12),
      );
      final end = Offset(
        center.dx + math.cos(angle) * radius,
        center.dy + math.sin(angle) * radius,
      );
      tickPaint.color = i < activeTicks ? Colors.greenAccent : Colors.white24;
      canvas.drawLine(start, end, tickPaint);
    }

    final borderPaint =
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = Colors.white54;
    canvas.drawCircle(center, radius, borderPaint);
  }

  @override
  bool shouldRepaint(covariant _FacePoseRingPainter oldDelegate) =>
      oldDelegate.progress != progress;
}

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
  bool _streaming = false;
  bool _processingFrame = false;
  bool _autoCaptureInFlight = false;
  int _poseIndex = 0;
  double _poseProgress = 0;
  String _poseMessage = 'Align your face inside the circle';
  final List<_PoseStep> _poseSequence = const [
    _PoseStep.center,
    _PoseStep.left,
    _PoseStep.right,
    _PoseStep.up,
  ];

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
      debugPrint('FaceRegister: initState');
    }
    _initialize();
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
      debugPrint('FaceRegister: _initialize (model + camera)');
    }

    try {
      await _embedder!.load();
      if (kDebugMode) {
        debugPrint('FaceRegister: model loaded successfully');
      }
      await Future.delayed(const Duration(milliseconds: 150));
      await _initCamera();
    } catch (e) {
      if (!mounted) return;
      if (kDebugMode) {
        debugPrint('FaceRegister: Initialization error: $e');
      }
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Initialization error: $e')));
    }
  }

  Future<void> _cleanUp() async {
    try {
      if (kDebugMode) {
        debugPrint('FaceRegister: _cleanUp');
      }
      await _stopPoseGuidanceStream();
      final cam = _cam;
      if (cam != null) {
        if (mounted) {
          setState(() {
            _cam = null;
            _lastImage = null;
          });
        } else {
          _cam = null;
          _lastImage = null;
        }
        await cam.dispose();
        if (kDebugMode) {
          debugPrint('FaceRegister: Camera disposed');
        }
      }

      if (_embedder != null) {
        _embedder!.close();
        _embedder = null;
        if (kDebugMode) {
          debugPrint('FaceRegister: Embedder disposed');
        }
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('FaceRegister: Cleanup error: $e');
      }
    }
  }

  void _resetPoseTracking({bool notify = true}) {
    if (!mounted) return;
    if (notify) {
      setState(() {
        _poseIndex = 0;
        _poseProgress = 0;
        _poseMessage = 'Align your face inside the circle';
      });
    } else {
      _poseIndex = 0;
      _poseProgress = 0;
      _poseMessage = 'Align your face inside the circle';
    }
  }

  Future<void> _stopPoseGuidanceStream() async {
    if (!_streaming || _cam == null) return;
    try {
      await _cam!.stopImageStream();
    } catch (e) {
      if (kDebugMode) {
        debugPrint('FaceRegister: stop stream error: $e');
      }
    } finally {
      _streaming = false;
      _processingFrame = false;
    }
  }

  Future<void> _startPoseGuidanceStream() async {
    final controller = _cam;
    if (controller == null || _streaming) return;

    try {
      _streaming = true;
      await controller.startImageStream((CameraImage image) async {
        if (_processingFrame || _autoCaptureInFlight) return;
        _processingFrame = true;
        try {
          final input = _cameraImageToInputImage(image, controller);
          if (input == null) return;
          final faces = await _detector.processImage(input);
          if (faces.isEmpty) {
            _resetPoseTracking();
            return;
          }
          final face = faces.first;
          if (_advancePose(face) && _poseIndex >= _poseSequence.length) {
            await _onPoseSequenceComplete();
          }
        } catch (e) {
          if (kDebugMode) {
            debugPrint('FaceRegister: pose stream error: $e');
          }
        } finally {
          _processingFrame = false;
        }
      });
    } catch (e) {
      _streaming = false;
      if (kDebugMode) {
        debugPrint('FaceRegister: start stream error: $e');
      }
    }
  }

  Future<void> _onPoseSequenceComplete() async {
    if (_autoCaptureInFlight || _cam == null) return;
    _autoCaptureInFlight = true;
    await _stopPoseGuidanceStream();
    try {
      await _captureAndDetect();
    } finally {
      _autoCaptureInFlight = false;
    }
  }

  bool _advancePose(Face face) {
    if (_poseIndex >= _poseSequence.length) {
      return false;
    }
    final yaw = face.headEulerAngleY ?? 0;
    final pitch = face.headEulerAngleX ?? 0;
    final step = _poseSequence[_poseIndex];
    bool satisfied = false;

    switch (step) {
      case _PoseStep.center:
        satisfied = yaw.abs() < 8 && pitch.abs() < 8;
        break;
      case _PoseStep.left:
        satisfied = yaw < -15;
        break;
      case _PoseStep.right:
        satisfied = yaw > 15;
        break;
      case _PoseStep.up:
        satisfied = pitch < -10;
        break;
    }

    if (satisfied) {
      _poseIndex++;
      final progress =
          (_poseIndex / _poseSequence.length).clamp(0.0, 1.0).toDouble();
      if (mounted) {
        setState(() {
          _poseProgress = progress;
          _poseMessage =
              _poseIndex >= _poseSequence.length
                  ? 'Hold still... capturing'
                  : _poseInstructionForStep(_poseSequence[_poseIndex]);
        });
      }
      return true;
    } else {
      if (mounted) {
        setState(() {
          _poseMessage = _poseInstructionForStep(step);
        });
      }
    }
    return false;
  }

  Uint8List _convertToNv21(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final int ySize = width * height;
    final int uvSize = width * height ~/ 2;
    final Uint8List bytes = Uint8List(ySize + uvSize);

    int offset = 0;
    for (int row = 0; row < height; row++) {
      final int rowStart = row * yPlane.bytesPerRow;
      bytes.setRange(offset, offset + width, yPlane.bytes, rowStart);
      offset += width;
    }

    final int chromaHeight = height ~/ 2;
    final int chromaWidth = width ~/ 2;
    final int uRowStride = uPlane.bytesPerRow;
    final int vRowStride = vPlane.bytesPerRow;
    final int uPixelStride = uPlane.bytesPerPixel ?? 1;
    final int vPixelStride = vPlane.bytesPerPixel ?? 1;

    for (int row = 0; row < chromaHeight; row++) {
      final int uRowStart = row * uRowStride;
      final int vRowStart = row * vRowStride;
      for (int col = 0; col < chromaWidth; col++) {
        final int uIndex = uRowStart + col * uPixelStride;
        final int vIndex = vRowStart + col * vPixelStride;
        bytes[offset++] = vPlane.bytes[vIndex];
        bytes[offset++] = uPlane.bytes[uIndex];
      }
    }

    return bytes;
  }

  InputImage? _cameraImageToInputImage(
    CameraImage image,
    CameraController controller,
  ) {
    InputImageFormat format;
    Uint8List bytes;
    int bytesPerRow;

    if (Platform.isIOS) {
      format = InputImageFormat.bgra8888;
      bytes = image.planes.first.bytes;
      bytesPerRow = image.planes.first.bytesPerRow;
    } else {
      format = InputImageFormat.nv21;
      bytes = _convertToNv21(image);
      bytesPerRow = image.planes.first.bytesPerRow;
    }

    final sensorOrientation = controller.description.sensorOrientation;
    InputImageRotation? rotation =
        InputImageRotationValue.fromRawValue(sensorOrientation) ??
        InputImageRotation.rotation0deg;

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: bytesPerRow,
      ),
    );
  }

  Future<void> _initCamera() async {
    if (_cam != null) {
      await _cam!.dispose();
      _cam = null;
    }

    try {
      if (kDebugMode) {
        debugPrint('FaceRegister: Initializing camera...');
      }
      final cams = await availableCameras();
      if (kDebugMode) {
        debugPrint('FaceRegister: Available cameras: ${cams.length}');
      }
      if (cams.isEmpty) {
        throw CameraException('No cameras', 'No cameras available on device');
      }

      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      if (kDebugMode) {
        debugPrint(
          'FaceRegister: Selected camera: ${front.name} (${front.lensDirection})',
        );
      }

      final ctrl = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      try {
        await ctrl.initialize();
      } on CameraException catch (e) {
        // Common case: camera still "in use" right after coming back
        if (kDebugMode) {
          debugPrint(
            'FaceRegister: CameraException on initialize: ${e.code} / $e',
          );
        }
        // Simple retry after a short delay
        if (e.code == 'CameraAccess' || e.code == 'CameraInUse') {
          await Future.delayed(const Duration(milliseconds: 300));
          if (!mounted) return;
          if (kDebugMode) {
            debugPrint('FaceRegister: retrying camera initialize...');
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
          debugPrint('FaceRegister: Camera initialized and locked');
        }
      } catch (e) {
        if (kDebugMode) {
          debugPrint('FaceRegister: Warning: Could not lock orientation: $e');
        }
      }

      if (!mounted) return;
      setState(() {
        _cam = ctrl;
        _busy = false;
        _resetPoseTracking(notify: false);
        _poseMessage = _poseInstructionForStep(_poseSequence.first);
      });
      await _startPoseGuidanceStream();
    } catch (e) {
      if (!mounted) return;
      if (kDebugMode) {
        debugPrint('FaceRegister: _initCamera error: $e');
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
        // print
        var decodedBody = jsonDecode(resp.body);

        if (decodedBody['error'] == 0) {
          // _login();
          debugPrint('Image has been saved successfully');
        } else {
          debugPrint('error is 1, error in daving the embedding');
        }
        if (kDebugMode) {
          debugPrint('✅ Embedding saved: ${resp.body}');
        }
      } else {
        if (kDebugMode) {
          debugPrint('❌ Server error ${resp.statusCode}: ${resp.body}');
        }
        throw Exception('Server ${resp.statusCode}');
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('❌ Network error: $e');
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
      _poseIndex = 0;
      _poseProgress = 0;
      _poseMessage = 'Align your face inside the circle';
    });
  }

  Future<void> _recapture() async {
    if (kDebugMode) {
      debugPrint('FaceRegister: re-capture');
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
      debugPrint('FaceRegister: login btn clicked');
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

      await _stopPoseGuidanceStream();

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
      _saveImage();
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
      debugPrint('FaceRegister: dispose');
    }

    _stopPoseGuidanceStream();
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
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Register Face')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              // children: [
              //   if (_lastImage != null) ...[
              //     ElevatedButton.icon(
              //       onPressed: _busy ? null : _recapture,
              //       icon: const Icon(Icons.camera_alt),
              //       label: Text(_busy ? 'Working…' : 'Re-Capture'),
              //     ),
              //     const SizedBox(width: 8),
              //     ElevatedButton.icon(
              //       onPressed: _busy ? null : _saveImage,
              //       icon: const Icon(Icons.save),
              //       label: Text(_busy ? 'Saving…' : 'Register Face'),
              //     ),
              //   ] else ...[
              //     ElevatedButton.icon(
              //       onPressed: _busy ? null : _captureAndDetect,
              //       icon: const Icon(Icons.camera_alt),
              //       label: Text(_busy ? 'Working…' : 'Capture'),
              //     ),
              //   ],
              //   const SizedBox(width: 8),
              //   ElevatedButton.icon(
              //     onPressed: _busy ? null : _login,
              //     icon: const Icon(Icons.login),
              //     label: Text(_busy ? 'Processing…' : 'Login'),
              //   ),
              // ],

            ),
          ),
          const SizedBox(height: 8),
          if (_lastImage == null)
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final previewWidth = cam.value.previewSize!.height;
                  final previewHeight = cam.value.previewSize!.width;
                  final overlaySize =
                      math.min(constraints.maxWidth, constraints.maxHeight) *
                      0.8;
                  return Container(
                    color: Colors.transparent, // Fills all remaining space
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        Positioned.fill(
                          child: Container(color: Colors.transparent),
                        ),
                        SizedBox(
                          width: overlaySize,
                          height: overlaySize,
                          child: ClipOval(
                            child: FittedBox(
                              fit: BoxFit.cover,
                              child: SizedBox(
                                width: previewWidth,
                                height: previewHeight,
                                child: CameraPreview(cam),
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          width: overlaySize,
                          height: overlaySize,
                          child: IgnorePointer(
                            child: CustomPaint(
                              painter: _FacePoseRingPainter(_poseProgress),
                            ),
                          ),
                        ),
                        Positioned(
                          bottom: 32,
                          left: 16,
                          right: 16,
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                'Move your head slowly to complete the circle.',
                                textAlign: TextAlign.center,
                                style: Theme.of(context).textTheme.titleMedium
                                    ?.copyWith(color: Colors.black),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                _poseMessage,
                                textAlign: TextAlign.center,
                                style: Theme.of(context).textTheme.bodyMedium
                                    ?.copyWith(color: Colors.black87),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  );
                },
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
                        AlertDialog(
                          title: const Text('Image Register Successfully'),
                          content: const Text(
                            'Image registration is successful! Please click Login. ',
                          ),
                          actions: <Widget>[
                            TextButton(
                              onPressed: () => Navigator.pop(context, 'Cancel'),
                              child: const Text('Cancel'),
                            ),
                            TextButton(
                              onPressed:
                                  () => Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) => const VerifyFace(),
                                    ),
                                  ),
                              child: const Text('Login'),
                            ),
                          ],
                        ),
                        // SizedBox(
                        //   width: displayW,
                        //   height: displayH,
                        //   child: Stack(
                        //     children: [
                        //       if (_uprightBytes != null)
                        //         Image.memory(
                        //           _uprightBytes!,
                        //           width: displayW,
                        //           height: displayH,
                        //           fit: BoxFit.contain,
                        //         )
                        //       else
                        //         const SizedBox.shrink(),
                        //       for (final face in _faces)
                        //         Positioned(
                        //           left: face.boundingBox.left * sx,
                        //           top: face.boundingBox.top * sy,
                        //           child: Container(
                        //             width: face.boundingBox.width * sx,
                        //             height: face.boundingBox.height * sy,
                        //             decoration: BoxDecoration(
                        //               border: Border.all(width: 2),
                        //             ),
                        //           ),
                        //         ),
                        //     ],
                        //   ),
                        // ),
                        // const SizedBox(height: 8),
                        // Padding(
                        //   padding: const EdgeInsets.symmetric(horizontal: 16.0),
                        //   child: Align(
                        //     alignment: Alignment.centerLeft,
                        //     child: Text(
                        //       'Detected faces: ${_faces.length}',
                        //       style: Theme.of(context).textTheme.titleMedium,
                        //     ),
                        //   ),
                        // ),
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
