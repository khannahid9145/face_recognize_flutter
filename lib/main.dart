import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
// import 'screens/face_register.dart';
import 'screens/face_auth_service.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  // Ensure plugin services are initialized
  WidgetsFlutterBinding.ensureInitialized();
  
  // Test camera initialization
  try {
    cameras = await availableCameras();
    debugPrint('Found ${cameras.length} cameras');
  } catch (e) {
    debugPrint('Error initializing cameras: $e');
  }
  
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const FaceAuthRegister(),
    );
  }
}


