#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

fetch() {
  local url="$1"
  local destination="$2"
  local expected="$3"
  mkdir -p "$(dirname "$destination")"
  if [[ ! -f "$destination" ]]; then
    curl -fL --retry 3 --connect-timeout 30 --max-time 900 \
      -o "${destination}.download" "$url"
    mv -f "${destination}.download" "$destination"
  fi
  local actual
  actual="$(shasum -a 256 "$destination" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $destination" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
}

SA="$DATA_DIR/spamassassin"
fetch "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2" \
  "$SA/20030228_easy_ham.tar.bz2" \
  "2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438"
fetch "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2" \
  "$SA/20030228_easy_ham_2.tar.bz2" \
  "b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38"
fetch "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2" \
  "$SA/20030228_hard_ham.tar.bz2" \
  "ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9"
fetch "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2" \
  "$SA/20030228_spam.tar.bz2" \
  "c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743"
fetch "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2" \
  "$SA/20030228_spam_2.tar.bz2" \
  "40bc444c596ff3e71505ae334f115b16f9b946786ee68f09b921b6b99217fd00"

for archive in "$SA"/20030228_*.tar.bz2; do
  tar -xjf "$archive" -C "$SA"
done

fetch "https://monkey.org/~jose/phishing/phishing3.mbox" \
  "$DATA_DIR/nazario/phishing3.mbox" \
  "b29336d2e31c2dff19639415e98ed2aced5b4d63675ea5e29bea1a7d4e452841"

PHISHING_POT="$DATA_DIR/phishing_pot"
PHISHING_POT_COMMIT="014392a2dfcdfae50a12841e9258b481888275a9"
if [[ ! -d "$PHISHING_POT/.git" ]]; then
  git init "$PHISHING_POT"
  git -C "$PHISHING_POT" remote add origin https://github.com/rf-peixoto/phishing_pot.git
fi
git -C "$PHISHING_POT" fetch --depth 1 origin "$PHISHING_POT_COMMIT"
git -C "$PHISHING_POT" checkout --detach "$PHISHING_POT_COMMIT"

ENRON="$DATA_DIR/enron-spam"
declare -a ENRON_HASHES=(
  "3d6a1fcced6c30b1701a3fd04c6b912d7046208f6ef6f4fdcf022f64bdc6c32f"
  "9b5bf5c4844eb4ecead52de8c676ff8761c2d7037fdc175851c88ef52c827315"
  "9321ca1c195b45b6f0fb795963a8020ff317e7ebac2c102205272f0b91a1ac0b"
  "61be71c59d7e9fdf99df63b814ddbeb18e0c9e8a69ce691d8e4e8127ed3588aa"
  "d7b839c2a7071b6d52f1d76be8d05e66733d872614475fad7522740d7c60540d"
  "283f70c45a7d819e0b988536fbb8e21cc17c17a22998403c2255d2c38cbfc5a0"
)
for index in 1 2 3 4 5 6; do
  archive="$ENRON/enron${index}.tar.gz"
  fetch "http://www2.aueb.gr/users/ion/data/enron-spam/raw/enron${index}.tar.gz" \
    "$archive" "${ENRON_HASHES[$((index - 1))]}"
  tar -xzf "$archive" -C "$ENRON"
done

echo "Pinned ConformalGuard source snapshots are ready in $DATA_DIR"
