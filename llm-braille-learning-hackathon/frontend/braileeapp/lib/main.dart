import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:flutter_tts/flutter_tts.dart';
import 'package:vibration/vibration.dart';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;
import 'package:flutter/foundation.dart'; // ✅ Add this line for listEquals


void main() => runApp(const BrailleApp());

class BrailleApp extends StatelessWidget {
  const BrailleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: BrailleHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class BrailleHome extends StatefulWidget {
  @override
  _BrailleHomeState createState() => _BrailleHomeState();
}

class _BrailleHomeState extends State<BrailleHome> {
  late stt.SpeechToText _speech;
  bool _isListening = false;
  String _spokenText = '';
  String _responseText = '';
  FlutterTts _tts = FlutterTts();
  late WebSocketChannel channel;

  Set<int> _userTappedDots = {};
  String _targetLetter = ''; // last word's last letter

  @override
  void initState() {
    super.initState();
    _speech = stt.SpeechToText();
    _tts.setLanguage("en-US");
    _tts.setPitch(1.0);

    channel = WebSocketChannel.connect(
      Uri.parse('ws://192.168.1.32:8000/ws/tutor'),
    );

    channel.stream.listen((response) {
      final decoded = jsonDecode(response);
      setState(() {
        _responseText = decoded['response'];
        _targetLetter = _extractLastLetter(_spokenText);
        _userTappedDots.clear(); // reset taps
      });
      _speak(_responseText);
    }, onError: (error) {
      print('WebSocket error: $error');
    });
  }

  String _extractLastLetter(String text) {
    final words = text.trim().split(RegExp(r'\s+'));
    if (words.isEmpty) return '';
    final lastWord = words.last.replaceAll(RegExp(r'[^\w]'), '');
    return lastWord.isNotEmpty ? lastWord.toLowerCase().substring(lastWord.length - 1) : '';
  }

  Future<void> _listen() async {
    bool available = await _speech.initialize();
    if (available) {
      setState(() => _isListening = true);
      _speech.listen(onResult: (result) {
        setState(() {
          _spokenText = result.recognizedWords;
        });
        _speech.stop();
        _sendToApi(_spokenText);
        setState(() => _isListening = false);
      });
    }
  }

  void _sendToApi(String text) {
    final message = jsonEncode({
      "user_id": "user123",
      "message": text,
      "level": "Beginner"
    });

    channel.sink.add(message);
  }

  Future<void> _speak(String text) async {
    await _tts.speak(text);
  }

  void _onDotTap(int dot) {
    setState(() {
      if (_userTappedDots.contains(dot)) {
        _userTappedDots.remove(dot);
      } else {
        _userTappedDots.add(dot);
      }
    });

    if (_targetLetter.isNotEmpty) {
      List<int> correctDots = LessonService.getBrailleDots(_targetLetter);
      correctDots.sort();
      List<int> tapped = _userTappedDots.toList()..sort();

      if (listEquals(tapped, correctDots)) {
        Vibration.vibrate(duration: 200);
      }
    }
  }

  Widget _buildBrailleDotGrid() {
    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: List.generate(6, (index) {
        int dotNumber = index + 1;
        bool isSelected = _userTappedDots.contains(dotNumber);
        return GestureDetector(
          onTap: () => _onDotTap(dotNumber),
          child: Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isSelected ? Colors.green : Colors.grey.shade400,
            ),
            child: Center(
              child: Text(
                '$dotNumber',
                style: const TextStyle(fontSize: 18, color: Colors.white),
              ),
            ),
          ),
        );
      }),
    );
  }

  @override
  void dispose() {
    channel.sink.close(status.goingAway);
    _speech.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Braille Voice App')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              ElevatedButton.icon(
                onPressed: _listen,
                icon: const Icon(Icons.mic),
                label: Text(_isListening ? 'Listening...' : 'Start Speaking'),
              ),
              const SizedBox(height: 20),
              Text('You said: $_spokenText'),
              const SizedBox(height: 20),
              Text('API Response: $_responseText'),
              const SizedBox(height: 20),
              if (_targetLetter.isNotEmpty)
                Text(
                  'Last letter: $_targetLetter — Tap correct Braille dots',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              const SizedBox(height: 10),
              _buildBrailleDotGrid(),
            ],
          ),
        ),
      ),
    );
  }
}

class LessonService {
  static List<int> getBrailleDots(String letter) {
    Map<String, List<int>> brailleMap = {
      'a': [1],
      'b': [1, 2],
      'c': [1, 4],
      'd': [1, 4, 5],
      'e': [1, 5],
      'f': [1, 2, 4],
      'g': [1, 2, 4, 5],
      'h': [1, 2, 5],
      'i': [2, 4],
      'j': [2, 4, 5],
      'k': [1, 3],
      'l': [1, 2, 3],
      'm': [1, 3, 4],
      'n': [1, 3, 4, 5],
      'o': [1, 3, 5],
      'p': [1, 2, 3, 4],
      'q': [1, 2, 3, 4, 5],
      'r': [1, 2, 3, 5],
      's': [2, 3, 4],
      't': [2, 3, 4, 5],
      'u': [1, 3, 6],
      'v': [1, 2, 3, 6],
      'w': [2, 4, 5, 6],
      'x': [1, 3, 4, 6],
      'y': [1, 3, 4, 5, 6],
      'z': [1, 3, 5, 6]
    };
    return brailleMap[letter.toLowerCase()] ?? [];
  }
}
