# Base Python image
FROM python:3.10-slim

# Set tensorflow log level to 3 to suppress warnings
ENV TF_CPP_MIN_LOG_LEVEL=3

# Set PyTorch CUDA allocation config to expandable segments
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set TensorFlow GPU allocator to cuda_malloc_async
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

# Set Hugging Face token
ENV HF_TOKEN=""

# Set timezone to America/Toronto
ENV TZ=America/Toronto

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    google-adk litellm \
    grpcio-status==1.48.2

# Install Python packages for GPU build
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0+cu118 

RUN pip install --no-cache-dir \
    tensorflow==2.12.0 \
    tensorflow-hub==0.12.0 \
    textattack==0.3.10 \
    nltk==3.9.1 \
    protobuf==3.20.3 \
    sentencepiece==0.1.99 \
    torchfile==0.1.0 \
    numpy==1.23.5 && \
    rm -rf /root/.cache/pip

# Download needed nltk dependencies
RUN python - <<'EOF'
import nltk, os, shutil, json, pickle, sys, time

NLTK_DATA_DIR='/usr/local/share/nltk_data'
src = os.path.join(NLTK_DATA_DIR, 'taggers', 'averaged_perceptron_tagger')
dst = os.path.join(NLTK_DATA_DIR, 'taggers', 'averaged_perceptron_tagger_eng')

for pkg in (
'punkt',
'averaged_perceptron_tagger',
'stopwords',
'wordnet',
'omw-1.4'
):
    nltk.download(pkg, download_dir=NLTK_DATA_DIR, force=True, raise_on_error=True)
    # Sleep for 1 second between downloads to avoid rate limiting
    time.sleep(1)
    
if not os.path.exists(os.path.join(src, 'averaged_perceptron_tagger.pickle')):
    sys.stderr.write(f"Error: averaged_perceptron_tagger.pickle not found in NLTK_DATA_DIR/taggers/averaged_perceptron_tagger\nExiting now.")
    sys.exit(1)

# Generate required files for nltk averaged_perceptron_tagger from the pickle file

with open(os.path.join(src, 'averaged_perceptron_tagger.pickle'), 'rb') as f:
    weights, tagdict, classes = pickle.load(f)
    classes = list(classes)

with open(os.path.join(src, 'averaged_perceptron_tagger.classes.json'), 'w') as f:
    json.dump(classes, f)

with open(os.path.join(src, 'averaged_perceptron_tagger.tagdict.json'), 'w') as f:
    json.dump(tagdict, f)

with open(os.path.join(src, 'averaged_perceptron_tagger.weights.json'), 'w') as f:
    json.dump(weights, f)

# duplicate and rename averaged_perceptron_tagger to averaged_perceptron_tagger_eng to account for old code in textattack that expects this

if os.path.isdir(src) and not os.path.isdir(dst):
    shutil.copytree(src, dst)
    for f in os.listdir(dst):
        if f.startswith('averaged_perceptron_tagger.'):
            os.rename(os.path.join(dst, f), os.path.join(dst, f.replace('averaged_perceptron_tagger.', 'averaged_perceptron_tagger_eng.')))
EOF

COPY ./multi_tool_agent /workspace/multi_tool_agent

# Install tzdata
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

# Expose the port that ADK will run on
EXPOSE 8000

# Set working dir & shell
WORKDIR /workspace
ENTRYPOINT ["adk", "api_server", "--host", "0.0.0.0", "--port", "8000"]

 # docker run --rm --gpus all -p 8000:8000 [image-name]