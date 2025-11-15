import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceContourPainter extends CustomPainter {
  final List<Face> faces;
  final int imageWidth;
  final int imageHeight;

  FaceContourPainter(this.faces, this.imageWidth, this.imageHeight);

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / imageWidth;
    final sy = size.height / imageHeight;

    final boxPaint = Paint()
      ..color = Colors.blue
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final pointPaint = Paint()
      ..color = Colors.blue
      ..style = PaintingStyle.fill;

    for (final face in faces) {
      // 1) Draw bounding box
      final rect = Rect.fromLTRB(
        face.boundingBox.left * sx,
        face.boundingBox.top * sy,
        face.boundingBox.right * sx,
        face.boundingBox.bottom * sy,
      );
      canvas.drawRect(rect, boxPaint);

      // 2) Draw contour points (eyes, lips, etc.)
      for (final contour in face.contours.values) {
        final points = contour?.points;
        if (points == null) continue;
        for (final p in points) {
          canvas.drawCircle(
            Offset(p.x * sx, p.y * sy),
            1.5, // radius
            pointPaint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
