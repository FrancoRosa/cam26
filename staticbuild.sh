#!/bin/bash

# Static Build for RKNN Applications
# Creates a Python environment bundle and .deb package for webcam-ins

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
CONDA_ENV="RKNN-Toolkit2"
PACKAGE_NAME="direct2"
BUILD_DIR="direct2-bundle"
RKNN_MODEL_ZOO_PY_UTILS="py_utils"

# Create a portable Python environment bundle
create_python_bundle() {
    log_info "Creating portable Python environment bundle..."
    
    # Check for required files
    if [[ ! -f "yolo11.rknn" ]]; then
        log_error "Required file yolo11.rknn not found in current directory"
        exit 1
    fi
    
    # Initialize conda
    eval "$(conda shell.bash hook)"
    conda activate $CONDA_ENV
    
    # Create bundle directory
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    mkdir -p "$BUILD_DIR/models"
    
    # Copy Python executable and essential libraries
    PYTHON_PREFIX=$(python -c "import sys; print(sys.prefix)")
    log_info "Python prefix: $PYTHON_PREFIX"
    
    # Create minimal Python runtime
    mkdir -p "$BUILD_DIR/python/bin"
    mkdir -p "$BUILD_DIR/python/lib"
    
    # Copy Python executable
    cp "$PYTHON_PREFIX/bin/python" "$BUILD_DIR/python/bin/" 2>/dev/null || true
    cp "$PYTHON_PREFIX/bin/python3" "$BUILD_DIR/python/bin/" 2>/dev/null || true
    
    # Copy essential Python libraries
    cp -r "$PYTHON_PREFIX/lib/python3.8" "$BUILD_DIR/python/lib/"
    
    # Copy RKNN libraries with their directory structure
    RKNN_SOURCE="$PYTHON_PREFIX/lib/python3.8/site-packages/rknn"
    if [[ -d "$RKNN_SOURCE" ]]; then
        cp -r "$RKNN_SOURCE" "$BUILD_DIR/python/lib/python3.8/site-packages/"
        log_success "RKNN libraries copied"
    else
        log_error "RKNN source not found: $RKNN_SOURCE"
        exit 1
    fi
    
    # Copy other essential packages
    PACKAGES=(
        "cv2"
        "numpy" 
        "PIL"
        "torch"
        "rknn_toolkit2"
        "py_utils"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        PKG_PATH="$PYTHON_PREFIX/lib/python3.8/site-packages/$pkg"
        if [[ -d "$PKG_PATH" ]]; then
            mkdir -p "$BUILD_DIR/python/lib/python3.8/site-packages/"
            cp -r "$PKG_PATH" "$BUILD_DIR/python/lib/python3.8/site-packages/"
            log_info "Copied package: $pkg"
        else
            # Try with different naming (e.g., cv2 -> opencv_python, py_utils -> py_utils)
            find "$PYTHON_PREFIX/lib/python3.8/site-packages" -name "*${pkg}*" -type d | head -1 | while read -r found_pkg; do
                if [[ -n "$found_pkg" ]]; then
                    cp -r "$found_pkg" "$BUILD_DIR/python/lib/python3.8/site-packages/"
                    log_info "Copied package: $(basename "$found_pkg")"
                else
                    log_warning "Package not found in site-packages: $pkg"
                fi
            done
        fi
    done
    
    # Copy py_utils from current directory or rknn_model_zoo
    if [[ -d "py_utils" ]]; then
        cp -r py_utils "$BUILD_DIR/python/lib/python3.8/site-packages/"
        log_info "Copied py_utils from current directory"
    elif [[ -d "$RKNN_MODEL_ZOO_PY_UTILS" ]]; then
        cp -r "$RKNN_MODEL_ZOO_PY_UTILS" "$BUILD_DIR/python/lib/python3.8/site-packages/py_utils/"
        log_info "Copied py_utils from rknn_model_zoo: $RKNN_MODEL_ZOO_PY_UTILS"
    else
        log_error "py_utils not found in current directory or $RKNN_MODEL_ZOO_PY_UTILS"
        exit 1
    fi
    
    # Copy application files
    cp direct2.py "$BUILD_DIR/"
    cp yolov11.py "$BUILD_DIR/"
    cp yolo11.rknn "$BUILD_DIR/models/"
    
    log_success "Python bundle created in: $BUILD_DIR"
}

# Create launcher script
create_launcher() {
    log_info "Creating launcher script..."
    
    cat > "$BUILD_DIR/webcam-ins" << 'EOF'
#!/bin/bash

# Webcam-ins launcher script with bundled Python environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up Python environment
export PYTHONPATH="$SCRIPT_DIR/python/lib/python3.8/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$SCRIPT_DIR/python/lib/python3.8/site-packages/rknn/api/lib/linux-aarch64:$LD_LIBRARY_PATH"

# Also add conda lib path if it exists
CONDA_LIB="/home/luckfox/miniforge3/envs/RKNN-Toolkit2/lib"
if [[ -d "$CONDA_LIB" ]]; then
    export LD_LIBRARY_PATH="$CONDA_LIB:$LD_LIBRARY_PATH"
fi

# Set Python executable
PYTHON_EXE="$SCRIPT_DIR/python/bin/python"

# If bundled python doesn't work, fall back to system python
if [[ ! -x "$PYTHON_EXE" ]]; then
    PYTHON_EXE="python3"
fi

# Change to script directory so relative paths work
cd "$SCRIPT_DIR"

# Run the application with explicit model path
exec "$PYTHON_EXE" direct2.py --model_path "$SCRIPT_DIR/yolo11.rknn" "$@"
EOF
    
    chmod +x "$BUILD_DIR/webcam-ins"
    log_success "Launcher script created"
}

# Create .deb package from bundle
create_bundle_deb() {
    log_info "Creating .deb package from bundle..."
    
    DEB_DIR="${PACKAGE_NAME}-bundle_1.0-1"
    rm -rf "$DEB_DIR" "${DEB_DIR}.deb"
    
    # Create package structure
    mkdir -p "$DEB_DIR/DEBIAN"
    mkdir -p "$DEB_DIR/usr/share/$PACKAGE_NAME/models"
    mkdir -p "$DEB_DIR/usr/bin"
    
    # Copy entire bundle
    cp -r "$BUILD_DIR"/* "$DEB_DIR/usr/share/$PACKAGE_NAME/"
    
    # Create symlink to launcher
    cat > "$DEB_DIR/usr/bin/$PACKAGE_NAME" << EOF
#!/bin/bash
exec /usr/share/$PACKAGE_NAME/webcam-ins "\$@"
EOF
    chmod +x "$DEB_DIR/usr/bin/$PACKAGE_NAME"
    
    # Create control file
    cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: $PACKAGE_NAME-bundle
Version: 1.0-1
Section: video
Priority: optional
Architecture: $(dpkg --print-architecture)
Depends: libc6, libstdc++6, libgcc-s1, libopencv-dev
Maintainer: YourName <your.email@example.com>
Description: YOLO11 Webcam Object Detection (Bundle Version)
 A computer vision application with bundled Python environment.
 This version includes all dependencies and should work without
 additional Python or conda installations.
EOF
    
    # Create install file
    cat > "$DEB_DIR/DEBIAN/${PACKAGE_NAME}-bundle.install" << EOF
usr/share/${PACKAGE_NAME}/*.py
usr/share/${PACKAGE_NAME}/models/*.rknn
usr/share/${PACKAGE_NAME}/python/*
usr/share/${PACKAGE_NAME}/webcam-ins
usr/bin/*
EOF
    
    # Build package
    dpkg-deb --build "$DEB_DIR"
    
    # Inspect package contents
    log_info "Inspecting .deb package contents:"
    dpkg -c "${DEB_DIR}.deb"
    
    log_success "Bundle package created: ${DEB_DIR}.deb"
}

# Test the bundle
test_bundle() {
    log_info "Testing the bundle..."
    
    if [[ ! -f "$BUILD_DIR/webcam-ins" ]]; then
        log_error "Bundle not found. Create it first."
        exit 1
    fi
    
    # Check for py_utils
    if [[ ! -d "$BUILD_DIR/python/lib/python3.8/site-packages/py_utils" ]]; then
        log_error "py_utils module not found in bundle!"
        ls -l "$BUILD_DIR/python/lib/python3.8/site-packages/"
        exit 1
    else
        log_info "py_utils found in bundle:"
        ls -l "$BUILD_DIR/python/lib/python3.8/site-packages/py_utils/"
    fi
    
    # Check for yolo11.rknn
    if [[ ! -f "$BUILD_DIR/models/yolo11.rknn" ]]; then
        log_error "yolo11.rknn not found in bundle!"
        ls -l "$BUILD_DIR/models/"
        exit 1
    else
        log_info "yolo11.rknn found in bundle:"
        ls -l "$BUILD_DIR/models/yolo11.rknn"
    fi
    
    # Test help command
    cd "$BUILD_DIR"
    if timeout 10s ./webcam-ins --help 2>&1; then
        log_success "Bundle test passed!"
    else
        log_warning "Bundle test failed or timed out"
        
        # Show what went wrong
        log_info "Debugging bundle..."
        echo "Python executable check:"
        ls -la python/bin/python* || echo "No python executable found"
        
        echo -e "\nRKNN library check:"
        find python/lib -name "*rknn*" -type d | head -5
        find python/lib -name "libcpputils2.so" 2>/dev/null || echo "libcpputils2.so not found in bundle"
        
        echo -e "\npy_utils check:"
        ls -la python/lib/python3.8/site-packages/py_utils || echo "py_utils not found"
        
        echo -e "\nyolo11.rknn check:"
        ls -la models/yolo11.rknn || echo "yolo11.rknn not found"
        
        echo -e "\nTrying direct python execution:"
        timeout 5s python/bin/python direct2.py --model_path yolo11.rknn --help 2>&1 || echo "Direct execution failed"
    fi
    
    cd ..
}

# Build complete bundle
build_complete_bundle() {
    log_info "Building complete bundle package..."
    
    create_python_bundle
    create_launcher
    test_bundle
    create_bundle_deb
    
    log_success "Complete bundle build finished!"
    echo
    echo "Files created:"
    echo "  Bundle directory: $BUILD_DIR/"
    echo "  Package file: ${PACKAGE_NAME}-bundle_1.0-1.deb"
    echo
    echo "To test:"
    echo "  cd $BUILD_DIR && ./webcam-ins --help"
}

# Cleanup broken installation
cleanup_broken() {
    log_info "Performing aggressive cleanup of broken webcam-ins installation..."
    
    # Kill any running processes
    log_info "Killing any running processes..."
    pkill -TERM -f webcam-ins 2>/dev/null || true
    sleep 2
    pkill -KILL -f webcam-ins 2>/dev/null || true
    
    # Remove all package-related files
    log_info "Removing all package-related files..."
    sudo rm -f /var/lib/dpkg/info/webcam-ins.* 2>/dev/null || true
    sudo rm -f /usr/bin/webcam-ins 2>/dev/null || true
    sudo rm -rf /usr/share/webcam-ins 2>/dev/null || true
    sudo rm -f /usr/share/applications/webcam-ins.desktop 2>/dev/null || true
    sudo rm -rf /usr/share/doc/webcam-ins 2>/dev/null || true
    
    # Remove package from dpkg database
    log_info "Removing package from dpkg database..."
    sudo dpkg --remove --force-all webcam-ins 2>/dev/null || true
    sudo dpkg --purge --force-all webcam-ins 2>/dev/null || true
    
    # Configure remaining packages
    log_info "Configuring remaining packages..."
    sudo dpkg --configure -a 2>/dev/null || true
    
    # Update package database
    log_info "Updating package database..."
    sudo apt-get update 2>/dev/null || true
    
    log_success "Nuclear cleanup completed!"
}

# Cleanup temporary files
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "$BUILD_DIR" "${PACKAGE_NAME}-bundle_1.0-1" "${PACKAGE_NAME}-bundle_1.0-1.deb" 2>/dev/null || true
    log_success "Cleanup completed"
}

# Install the package
install_package() {
    log_info "Installing package..."
    if sudo dpkg -i "${PACKAGE_NAME}-bundle_1.0-1.deb" 2>&1; then
        log_success "Package installed successfully!"
        sudo apt-get install -f -y 2>/dev/null || log_warning "Failed to fix dependencies"
    else
        log_error "Package installation failed!"
        echo "Try fixing dependencies with: sudo apt-get install -f"
    fi
}

# Main execution
main() {
    log_info "Starting static build for $PACKAGE_NAME"
    log_info "=================================================="
    
    # Parse command line arguments
    CLEANUP_BROKEN=false
    if [[ "$1" == "--cleanup-broken" ]]; then
        CLEANUP_BROKEN=true
    elif [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --cleanup-broken  Clean up broken webcam-ins installation state before building"
        echo "  --help, -h        Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Build and install complete bundle package"
        echo "  $0 --cleanup-broken   # Clean up broken installation, then build and install"
        exit 0
    elif [[ -n "$1" ]]; then
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
    fi
    
    # Run cleanup_broken if specified
    if [[ "$CLEANUP_BROKEN" == true ]]; then
        cleanup_broken
    fi
    
    # Run the complete bundle build
    build_complete_bundle
    
    # Prompt for installation
    read -p "Do you want to install the package? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        install_package
    else
        log_info "Skipping installation"
    fi
    
    # Prompt for cleanup
    read -p "Do you want to clean up temporary files? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        cleanup
    else
        log_info "Skipping cleanup"
    fi
    
    log_success "=================================================="
    log_success "All done! 🎉"
}

# Error handler
error_handler() {
    log_error "Script failed at line $1"
    cleanup
    exit 1
}

# Set error trap
trap 'error_handler $LINENO' ERR

# Run main function
main "$@"
