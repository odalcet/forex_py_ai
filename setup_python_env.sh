    #!/bin/bash

# Define your project environment name
ENV_NAME="tf-env"
PYTHON_VERSION="3.11"

echo "🧼 Cleaning up the past..."
sudo dnf5 install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-devel gcc make automake autoconf libtool curl

echo "🐍 Creating virtual environment..."
python${PYTHON_VERSION} -m venv ${ENV_NAME}
source ${ENV_NAME}/bin/activate

echo "📦 Upgrading pip and installing base packages..."
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn tensorflow==2.15

echo "📥 Downloading TA-Lib C library..."
cd /tmp
curl -L -O https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install

echo "🐍 Installing TA-Lib Python wrapper..."
pip install TA-Lib==0.4.0

echo "✅ Verifying TA-Lib import..."
python -c "import talib; print('TA-Lib version:', talib.__version__)"

echo "🧠 Making VS Code use this environment..."
mkdir -p .vscode
cat > .vscode/settings.json <<EOF
{
    "python.defaultInterpreterPath": "\${workspaceFolder}/${ENV_NAME}/bin/python"
}
EOF

echo "🎉 All done! Activate your environment with:"
echo "source ${ENV_NAME}/bin/activate"
