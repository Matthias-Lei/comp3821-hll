
REPO_URL="https://github.com/Matthias-Lei/comp3821-hll-data.git"

REPO_DIR="/tmp/cebs-repository"
SOURCE_DIR="$(pwd)/data"
STREAMS_DIR="$SOURCE_DIR/streams"
CARDINALITY_FILE="$SOURCE_DIR/cardinalities.tsv"
GIT_USERNAME="cat"
GIT_EMAIL="meow@example.com"

sudo apt-get update
sudo apt-get install -y rsync git-lfs


cd python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make
python3 create_test_streams.py
python3 demo.py
python3 compute_cardinalities.py
cd ..

htop
df /workspaces -h
ls -lhS data/streams


unset GITHUB_TOKEN
gh auth login
gh repo list --limit 4
gh auth setup-git

git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

# git lfs version
# git lfs install --local
# git lfs track "data/streams/*.cebs"

mkdir -p data/streams
rsync -a -P --exclude="*_m10000.cebs" "$STREAMS_DIR"/ "$REPO_DIR"/data/streams/
rsync -a "$CARDINALITY_FILE" data/
rsync -a "$CARDINALITY_FILE" "$REPO_DIR"/data/
tree "$REPO_DIR"

cd "$REPO_DIR"
git add .
git config user.name "$GIT_USERNAME"
git config user.email "$GIT_EMAIL"
# git commit -m "Add CEBS streams and cardinalities"
git commit --amend -m "Update CEBS streams and cardinalities as at $(date +%Y-%m-%dT%H:%M:%S)"
git push origin --force
