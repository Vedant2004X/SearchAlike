from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging
import sys
import threading
import webview
import atexit
import time

import config
from src.feature_extractor import FeatureExtractor
from src.similarity_search import SimilaritySearch
from src.utils import allowed_file, save_uploaded_file, get_image_paths, clean_upload_folder
from src.preprocessing import load_and_validate_image

import io

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYINSTALLER RESOURCE PATH HELPER
# ============================================================================


def resource_path(relative_path):
    """Get absolute path to resource, works for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


TEMPLATE_DIR = resource_path("templates")
STATIC_DIR = resource_path("static")


# ============================================================================
# SINGLE INSTANCE LOCK - FIXED VERSION
# ============================================================================

def get_lock_file():
    """Get lock file path - use app folder instead of TEMP"""
    return config.BASE_DIR / ".searchalike.lock"


def cleanup_lock():
    """Clean up lock file on exit"""
    try:
        lock_file = get_lock_file()
        if lock_file.exists():
            lock_file.unlink()
            logger.info("Lock file cleaned up")
    except Exception as e:
        logger.warning(f"Could not clean up lock file: {e}")


def check_single_instance():
    """
    Check if another instance is already running.
    Returns True if we should continue, False if another instance exists.
    """
    lock_file = get_lock_file()

    # If lock file exists, check if process is still running
    if lock_file.exists():
        try:
            with open(lock_file, 'r') as f:
                old_pid = f.read().strip()

            # Try to check if process with that PID is running
            if sys.platform == 'win32':
                try:
                    import psutil
                    if psutil.pid_exists(int(old_pid)):
                        logger.warning(f"SearchAlike already running (PID: {old_pid})")
                        return False
                except:
                    # If we can't check, assume process is gone
                    pass
            else:
                # On Linux/Mac, try to send signal 0 (check if process exists)
                try:
                    os.kill(int(old_pid), 0)
                    logger.warning(f"SearchAlike already running (PID: {old_pid})")
                    return False
                except (OSError, ValueError):
                    # Process not running, clean up lock
                    lock_file.unlink()
        except Exception as e:
            logger.warning(f"Could not read lock file: {e}")
            # Clean up corrupted lock file
            try:
                lock_file.unlink()
            except:
                pass

    # Create new lock file with current PID
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"Lock file created: {lock_file}")
        return True
    except Exception as e:
        logger.error(f"Could not create lock file: {e}")
        return True  # Continue anyway


# ============================================================================
# INITIALIZE FLASK APP
# ============================================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

feature_extractor = None
similarity_search = None
index_loaded = False


# ============================================================================
# MODEL INITIALIZATION FUNCTIONS
# ============================================================================


def check_and_train_if_needed():
    """
    Check if model needs training and train if necessary.
    Returns True if ready to use, False otherwise.
    """
    global feature_extractor, similarity_search, index_loaded

    logger.info("=" * 70)
    logger.info("   CHECKING MODEL STATUS")
    logger.info("=" * 70)

    # Check if trained model exists
    if config.TRIPLET_MODEL_PATH.exists():
        logger.info("   [OK] Trained triplet model found!")
        logger.info(f"   Model: {config.TRIPLET_MODEL_PATH}")
        return True

    # Model doesn't exist - check if we should auto-train
    logger.info("    [WARNING] No trained model found")

    if not config.AUTO_TRAIN_ON_FIRST_RUN:
        logger.warning("   Auto-train disabled. Please run: python train_triplet.py")
        return False

    # Check training requirements
    ready, messages = config.check_training_requirements()

    for msg in messages:
        logger.info(msg)

    if not ready:
        logger.error("   [ERROR] Cannot train - requirements not met")
        logger.info("   To fix:")
        logger.info(f"   1. Add images to: {config.IMAGES_FOLDER}")
        logger.info(f"   2. Organize in subfolders by category")
        logger.info(f"   3. Minimum {config.MIN_IMAGES_FOR_TRAINING} images required")
        return False

    # Requirements met - start training
    logger.info("=" * 70)
    logger.info("  STARTING AUTO-TRAINING")
    logger.info("=" * 70)
    logger.info("   This will take some time... Please wait.")
    logger.info("")

    try:
        # Import training modules
        from src.triplet_trainer import TripletTrainer, get_default_transforms, freeze_backbone
        from src.triplet_network import TripletNetwork, OnlineTripletLoss
        from src.triplet_dataset import create_triplet_dataloader, generate_labels_from_folders
        from sklearn.model_selection import train_test_split
        import torch

        # Load data
        logger.info("   Loading images...")
        image_paths, labels = generate_labels_from_folders(config.IMAGES_FOLDER)

        # Convert to numeric labels
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        numeric_labels = [label_to_idx[label] for label in labels]

        logger.info(f"   Found {len(image_paths)} images")
        logger.info(f"   Classes: {len(set(labels))}")

        # Train/val split
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, numeric_labels,
            test_size=config.VAL_SPLIT,
            stratify=numeric_labels,
            random_state=42
        )

        logger.info(f"   Train: {len(train_paths)}, Val: {len(val_paths)}")

        # Create transforms
        train_transform = get_default_transforms(augment=True)
        val_transform = get_default_transforms(augment=False)

        # Create data loaders
        logger.info("   Creating data loaders...")
        train_loader = create_triplet_dataloader(
            train_paths, train_labels, train_transform,
            batch_size=config.TRAIN_BATCH_SIZE,
            mode=config.TRIPLET_MINING_MODE,
            shuffle=True,
            num_workers=config.NUM_WORKERS
        )

        val_loader = create_triplet_dataloader(
            val_paths, val_labels, val_transform,
            batch_size=config.TRAIN_BATCH_SIZE,
            mode=config.TRIPLET_MINING_MODE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        # Create model
        logger.info("    Creating model...")
        model = TripletNetwork(
            embedding_dim=config.TRIPLET_EMBEDDING_DIM,
            pretrained=True,
            backbone=config.TRIPLET_BACKBONE
        )

        if config.FREEZE_BACKBONE_INITIALLY:
            freeze_backbone(model, freeze=True)

        # Create loss and optimizer
        loss_fn = OnlineTripletLoss(
            margin=config.TRIPLET_MARGIN,
            triplet_selector=config.TRIPLET_LOSS_TYPE
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.TRAIN_LEARNING_RATE,
            weight_decay=config.TRAIN_WEIGHT_DECAY
        )

        # Create trainer
        logger.info("   Initializing trainer...")
        trainer = TripletTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=config.DEVICE,
            checkpoint_dir=str(config.CHECKPOINT_FOLDER),
            log_dir=str(config.LOG_DIR)
        )

        # Train
        logger.info("=" * 70)
        logger.info(f"  TRAINING FOR {config.TRAIN_EPOCHS} EPOCHS")
        logger.info("=" * 70)

        trainer.train(
            num_epochs=config.TRAIN_EPOCHS,
            save_every=config.SAVE_CHECKPOINT_EVERY,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE
        )

        logger.info("=" * 70)
        logger.info("   [OK] TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"   Model saved: {config.TRIPLET_MODEL_PATH}")
        logger.info("")

        # Update config to use triplet model
        config.USE_TRIPLET_MODEL = True

        return True

    except Exception as e:
        logger.error(f"   [ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def initialize_models():
    """Initialize feature extractor and similarity search"""
    global feature_extractor, similarity_search, index_loaded

    # Determine which model to use
    triplet_model_path = None
    embeddings_file = config.EMBEDDINGS_FILE
    index_file = config.FAISS_INDEX_FILE

    if config.USE_TRIPLET_MODEL and config.TRIPLET_MODEL_PATH.exists():
        triplet_model_path = str(config.TRIPLET_MODEL_PATH)
        embeddings_file = config.TRIPLET_EMBEDDINGS_FILE
        index_file = config.TRIPLET_FAISS_INDEX
        logger.info(f"   [OK] Using trained Triplet Network")
        logger.info(f"   Model: {triplet_model_path}")
    else:
        logger.info(f"   [INFO] Using pretrained {config.MODEL_NAME}")

    # Initialize feature extractor
    try:
        feature_extractor = FeatureExtractor(triplet_model_path=triplet_model_path)
    except Exception as e:
        logger.error(f"Failed to initialize feature extractor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    similarity_search = SimilaritySearch(use_faiss=True)

    # Try to load existing index
    try:
        if (index_file.exists() and
                config.IMAGE_PATHS_FILE.exists() and
                embeddings_file.exists()):
            logger.info("   [INFO] Loading existing search index...")
            similarity_search.load_index(
                index_path=index_file,
                paths_path=config.IMAGE_PATHS_FILE,
                features_path=embeddings_file
            )
            index_loaded = True
            logger.info("   [OK] Search index loaded successfully")
            return True

    except Exception as e:
        logger.warning(f"    Could not load existing index: {e}")
        logger.info("   Will build new index...")

    # Build new index
    logger.info("   [INFO] Building new search index...")

    try:
        image_paths = get_image_paths(config.IMAGES_FOLDER)

        if len(image_paths) == 0:
            logger.warning("    [WARNING] No images found in dataset folder")
            logger.warning(f"    Expected folder: {config.IMAGES_FOLDER}")
            logger.warning(f"    Please add images to: {config.IMAGES_FOLDER}")
            return False

        logger.info(f"  [INFO] Extracting features from {len(image_paths)} images...")
        features = feature_extractor.extract_features_batch(image_paths)

        logger.info("   [INFO] Building search index...")
        similarity_search.build_index(features, image_paths)

        logger.info("   [INFO] Saving index...")

        # Ensure embeddings folder exists
        config.EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)

        similarity_search.save_index(
            index_path=index_file,
            paths_path=config.IMAGE_PATHS_FILE
        )

        feature_extractor.save_features(features, embeddings_file)

        index_loaded = True
        logger.info(f"   [OK] Search index built for {len(image_paths)} images")
        return True

    except Exception as e:
        logger.error(f"   [ERROR] Error building index: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# FLASK ROUTES
# ============================================================================


@app.route('/')
def index():
    """Home page with upload form"""
    stats = similarity_search.get_statistics() if index_loaded else None
    model_info = feature_extractor.get_model_info() if feature_extractor else None
    return render_template('index.html', stats=stats, index_loaded=index_loaded, model_info=model_info)


@app.route('/search', methods=['POST'])
def search():
    """Handle image upload and search"""
    logger.info("=" * 50)
    logger.info("[SEARCH] NEW SEARCH REQUEST RECEIVED")
    logger.info("=" * 50)

    if not index_loaded:
        logger.error("Search index not initialized")
        flash('Search index not initialized. Please wait or check logs.', 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        logger.error("No file in request")
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    logger.info(f"File received: {file.filename}")

    if file.filename == '':
        logger.error("Empty filename")
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            logger.info("[SEARCH] Saving uploaded file...")
            relative_path = save_uploaded_file(file)

            if relative_path is None:
                raise RuntimeError("Failed to save uploaded file")

            # Convert to absolute path for processing
            abs_filepath = config.BASE_DIR / relative_path

            if not abs_filepath.exists():
                logger.error(f"File not found after saving: {abs_filepath}")
                flash('Error saving file', 'error')
                return redirect(url_for('index'))

            logger.info(f"[SEARCH] File saved to: {abs_filepath}")

            logger.info("[SEARCH] Validating image...")
            load_and_validate_image(str(abs_filepath))
            logger.info("[SEARCH] Image validated successfully")

            logger.info(f"[SEARCH] Extracting features from {abs_filepath}")
            query_features = feature_extractor.extract_features(str(abs_filepath))
            logger.info(f"[SEARCH] Features extracted. Shape: {query_features.shape}")

            top_k = int(request.form.get('top_k', config.TOP_K))
            top_k = min(max(top_k, 1), 50)
            logger.info(f"[SEARCH] Searching for top {top_k} similar images")

            results = similarity_search.search(query_features, top_k=top_k)
            logger.info(f"[SEARCH] Search completed. Found {len(results)} results")

            if len(results) > 0:
                logger.info("[SEARCH] --- Top Similarity Scores ---")
                for idx, (path, score) in enumerate(results[:5]):
                    logger.info(f"[SEARCH]   Top {idx + 1}: score={score:.4f} ({score * 100:.2f}%)")

            # Get query image filename only
            query_image_filename = Path(abs_filepath).name
            logger.info(f"[SEARCH] Query image filename: {query_image_filename}")

            # Build results with proper URL paths
            similar_images = []

            if len(results) == 0:
                logger.warning("[SEARCH] No similar images found")
            else:
                for img_path, score in results:
                    img_path_obj = Path(img_path)

                    logger.info(f"[SEARCH] Processing result: {img_path}")
                    logger.info(f"[SEARCH]   Image path object: {img_path_obj}")
                    logger.info(f"[SEARCH]   Image exists: {img_path_obj.exists()}")

                    # Get the relative path from DATA_FOLDER
                    try:
                        # Convert to absolute path if it's relative
                        if not img_path_obj.is_absolute():
                            abs_img_path = config.BASE_DIR / img_path_obj
                        else:
                            abs_img_path = img_path_obj

                        logger.info(f"[SEARCH]   Absolute path: {abs_img_path}")
                        logger.info(f"[SEARCH]   File exists: {abs_img_path.exists()}")

                        # Make relative to DATA_FOLDER
                        rel_path = abs_img_path.relative_to(config.DATA_FOLDER)
                        rel_path_str = str(rel_path).replace("\\", "/")

                        logger.info(f"[SEARCH]   Relative path: {rel_path_str}")

                        similar_images.append({
                            "path": url_for("serve_data_file", filename=rel_path_str),
                            "score": round(score * 100, 2),
                            "filename": abs_img_path.name
                        })

                        logger.info(f"[SEARCH]   Result URL: {url_for('serve_data_file', filename=rel_path_str)}")

                    except ValueError as e:
                        logger.warning(f"[SEARCH]   Could not make relative path: {e}")
                        logger.warning(f"[SEARCH]   Skipping result: {img_path}")
                        continue

            logger.info(f"[SEARCH] Rendering result page with {len(similar_images)} images")
            logger.info("=" * 50)

            return render_template(
                'result.html',
                query_image=query_image_filename,
                similar_images=similar_images,
                num_results=len(similar_images),
                model_info=feature_extractor.get_model_info()
            )

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(error_trace)

            return render_template(
                "error.html",
                error_code="Search Failed",
                error_message=str(e),
                traceback=error_trace
            )

    else:
        logger.error(f"Invalid file type: {file.filename}")
        flash('Invalid file type. Allowed types: ' + ', '.join(config.ALLOWED_EXTENSIONS), 'error')
        return redirect(url_for('index'))


@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """Serve files from data directory"""
    try:
        logger.info(f"[SERVE] Serving data file: {filename}")

        # Ensure filename doesn't contain path traversal attempts
        safe_path = Path(filename)
        if ".." in safe_path.parts:
            logger.error(f"[SERVE] Path traversal attempt: {filename}")
            return "Access denied", 403

        # Log the full path being served
        full_path = config.DATA_FOLDER / filename
        logger.info(f"[SERVE] Full path: {full_path}")
        logger.info(f"[SERVE] File exists: {full_path.exists()}")

        return send_from_directory(str(config.DATA_FOLDER), filename)
    except Exception as e:
        logger.error(f"[SERVE] Error serving file {filename}: {str(e)}")
        import traceback
        logger.error(f"[SERVE] Traceback: {traceback.format_exc()}")
        return "File not found", 404


@app.route('/rebuild-index', methods=['POST'])
def rebuild_index():
    """Rebuild the search index"""
    try:
        logger.info("[REBUILD] Rebuilding search index...")
        success = initialize_models()

        if success:
            flash('Search index rebuilt successfully', 'success')
        else:
            flash('Error rebuilding search index', 'error')

    except Exception as e:
        logger.error(f"[REBUILD] Error rebuilding index: {str(e)}")
        flash(f'Error: {str(e)}', 'error')

    return redirect(url_for('index'))


@app.route('/stats')
def stats():
    """Get search index statistics as JSON"""
    if index_loaded:
        stats = similarity_search.get_statistics()
        stats['model_info'] = feature_extractor.get_model_info()
        return jsonify(stats)
    else:
        return jsonify({'error': 'Index not loaded'}), 503


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'index_loaded': index_loaded,
        'model_type': feature_extractor.get_model_info()['model_type'] if feature_extractor else 'none'
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('error.html',
                           error_code=404,
                           error_message='Page not found'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return render_template('error.html',
                           error_code=500,
                           error_message='Internal server error'), 500


# ============================================================================
# STARTUP SEQUENCE
# ============================================================================


def startup_sequence():
    """Initialize application on startup"""
    print("=" * 70)
    print("  SEARCHALIKE - STARTUP SEQUENCE")
    print("=" * 70)
    print("startup_sequence entered")
    logger.info("Starting application startup sequence")

    # Ensure all required directories exist
    config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    config.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    config.IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    config.EMBEDDINGS_FOLDER.mkdir(parents=True, exist_ok=True)
    config.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Base directory: {config.BASE_DIR}")
    logger.info(f"Images folder: {config.IMAGES_FOLDER}")
    logger.info(f"Embeddings folder: {config.EMBEDDINGS_FOLDER}")

    # Clean old uploads
    clean_upload_folder()

    # Check for auto-training
    if not config.USE_TRIPLET_MODEL:
        if config.AUTO_TRAIN_ON_FIRST_RUN:
            logger.info("Checking if model training is needed...")
            check_and_train_if_needed()

    # Initialize models and build index
    success = initialize_models()
    if not success:
        logger.error("[ERROR] Failed to initialize models or index")
        logger.error("Please ensure images are in: " + str(config.IMAGES_FOLDER))
        sys.exit(1)

    logger.info("[OK] Startup sequence completed successfully")
    print("startup_sequence completed")
    print("=" * 70)


# ============================================================================
# FLASK START FUNCTION
# ============================================================================

def start_flask():
    """Start Flask server in background thread"""
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    import threading
    import time

    # ========== CHECK SINGLE INSTANCE ==========
    print("\n" + "=" * 70)
    print("  CHECKING SINGLE INSTANCE LOCK")
    print("=" * 70)

    if not check_single_instance():
        print("\n[ERROR] SearchAlike is already running!")
        print("   Only one instance can run at a time.")
        print("   Please close the existing window and try again.\n")
        sys.exit(0)

    print("[OK] Single instance check passed\n")

    # Register cleanup function to run on exit
    atexit.register(cleanup_lock)

    # ========== RUN STARTUP SEQUENCE ==========
    print("\n" + "=" * 70)
    print("  INITIALIZING APPLICATION")
    print("=" * 70 + "\n")

    startup_sequence()

    # ========== START FLASK & WEBVIEW ==========
    print("\n" + "=" * 70)
    print("  STARTING WEB SERVER & INTERFACE")
    print("=" * 70 + "\n")

    # Start Flask in background thread
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Wait for Flask to start
    time.sleep(2)

    print("[OK] Flask server started on http://127.0.0.1:5000")
    print("[INFO] Opening SearchAlike interface...\n")

    try:
        # Create and start WebView window
        window = webview.create_window(
            "SearchAlike - Image Similarity Search",
            "http://127.0.0.1:5000",
            width=1200,
            height=800
        )

        print("=" * 70)
        print("  APPLICATION READY!")
        print("=" * 70)
        print("\n[OK] SearchAlike is now running!")
        print("   Upload an image to find similar images in your dataset.\n")

        # Start WebView (blocks until window is closed)
        webview.start()

    except Exception as e:
        logger.error(f"Error starting WebView: {str(e)}")
        print(f"\n[ERROR] Error starting interface: {e}")
        print("   You can still access the app at http://127.0.0.1:5000\n")

    finally:
        print("\n" + "=" * 70)
        print("  SHUTTING DOWN SEARCHALIKE")
        print("=" * 70)
        cleanup_lock()
        logger.info("Application shutdown complete")
        print("[OK] Goodbye!\n")