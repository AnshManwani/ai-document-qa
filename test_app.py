from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        logger.info("Upload endpoint called")
        
        if 'document' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['document']
        logger.info(f"File received: {file.filename}")
        
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        # Simple file validation
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Please upload PDF files only for testing'}), 400

        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Check if file was saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            logger.info(f"File saved successfully. Size: {file_size} bytes")
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'message': 'File uploaded successfully!'
            })
        else:
            logger.error("File was not saved")
            return jsonify({'error': 'File upload failed'}), 500
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/chat')
def chat():
    return render_template('chat.html', filename="Test Document")

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_question = request.json.get('question', '').strip()
        
        if not user_question:
            return jsonify({'answer': 'Please enter a question'}), 400
        
        return jsonify({
            'answer': f'I received your question: "{user_question}". This is a test response from the server.'
        })
    
    except Exception as e:
        logger.error(f"Ask error: {str(e)}")
        return jsonify({'answer': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting TEST Flask application...")
    print("Upload folder:", app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5000)
