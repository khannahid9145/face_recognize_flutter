// lib/ml/face_embedder.dart
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// FaceEmbedder
/// - Call `await load()` once before using `run()`.
/// - Input: a cropped 112x112 RGB face image (img.Image)
/// - Output: 128-D normalized embedding (Float32List)
class FaceEmbedder {
  Interpreter? _interpreter;

  final List<int> inputShape;   // e.g. [1,112,112,3]
  final int embeddingSize;      // e.g. 128
  final String assetFilename;   // path in assets/

  FaceEmbedder({
    required this.inputShape,
    required this.embeddingSize,
    this.assetFilename = 'assets/face_embedding.tflite',
  });

  bool get isReady => _interpreter != null;

  /// Load the model from assets (must be awaited before run()).
  Future<void> load({int threads = 4}) async {
    if (_interpreter != null) return;
    try {
      final opts = InterpreterOptions()..threads = threads;
      _interpreter = await Interpreter.fromAsset(assetFilename, options: opts);
      print('FaceEmbedder model loaded: $assetFilename');
    } catch (e) {
      print('Error loading FaceEmbedder model: $e');
      rethrow;
    }
  }

  /// Free interpreter resources.
  void close() {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Run inference on a 112x112 face crop.
  Float32List run(img.Image face112) {
    final itp = _interpreter;
    if (itp == null) {
      throw StateError('Interpreter not initialized. Call await load() first.');
    }

    final h = inputShape[1];
    final w = inputShape[2];

    // Ensure the image matches model size
    if (face112.width != w || face112.height != h) {
      face112 = img.copyResize(face112, width: w, height: h);
    }

    final input = _imageToInput(face112, h, w);
    final output = List.generate(1, (_) => List.filled(embeddingSize, 0.0));

    itp.run(input, output);

    final raw = output[0].map((x) => x.toDouble()).toList(growable: false);
    return _l2Normalize(Float32List.fromList(raw));
  }

  /// Convert image into float32 tensor [1,h,w,3] normalized to [-1, 1].
  static List<List<List<List<double>>>> _imageToInput(
      img.Image im, int h, int w) {
    final input = List<List<List<List<double>>>>.generate(
      1,
      (_) => List<List<List<double>>>.generate(
        h,
        (_) => List<List<double>>.generate(
          w,
          (_) => List<double>.filled(3, 0.0),
        ),
      ),
    );

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = im.getPixel(x, y);
        final r = pixel.r.toDouble();
        final g = pixel.g.toDouble();
        final b = pixel.b.toDouble();

        // Normalize to [-1,1]
        input[0][y][x][0] = (r / 127.5) - 1.0;
        input[0][y][x][1] = (g / 127.5) - 1.0;
        input[0][y][x][2] = (b / 127.5) - 1.0;
      }
    }
    return input;
  }

  /// Normalize vector to unit length.
  static Float32List _l2Normalize(Float32List v) {
    double sumSq = 0.0;
    for (final x in v) sumSq += x * x;
    final norm = math.sqrt(sumSq);
    if (norm == 0.0) return Float32List(v.length);
    final out = Float32List(v.length);
    for (int i = 0; i < v.length; i++) {
      out[i] = v[i] / norm;
    }
    return out;
  }
}

/// Cosine similarity between two embeddings in range [-1,1].
double cosineSimilarity(Float32List a, Float32List b) {
  assert(a.length == b.length);
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int i = 0; i < a.length; i++) {
    final ai = a[i], bi = b[i];
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  final denom = math.sqrt(na) * math.sqrt(nb);
  return denom == 0.0 ? 0.0 : dot / denom;
}
