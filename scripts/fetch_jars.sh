#!/usr/bin/env bash
# Downloads SourceAFIS 3.18.1 and its runtime dependencies into ../jars.
set -euo pipefail
mkdir -p "$(dirname "$0")/../jars"
cd "$(dirname "$0")/../jars"
base="https://repo1.maven.org/maven2"
gavs=(
  "com/machinezoo/sourceafis/sourceafis/3.18.1"
  "com/machinezoo/stagean/stagean/1.3.0"
  "com/machinezoo/closeablescope/closeablescope/1.0.1"
  "com/machinezoo/noexception/noexception/1.9.1"
  "com/machinezoo/fingerprintio/fingerprintio/1.3.1"
  "it/unimi/dsi/fastutil/8.5.12"
  "commons-io/commons-io/2.15.0"
  "com/google/code/gson/gson/2.10.1"
  "com/fasterxml/jackson/core/jackson-databind/2.15.3"
  "com/fasterxml/jackson/core/jackson-core/2.15.3"
  "com/fasterxml/jackson/core/jackson-annotations/2.15.3"
  "com/fasterxml/jackson/dataformat/jackson-dataformat-cbor/2.15.3"
  "com/github/mhshams/jnbis/2.1.2"
  "org/slf4j/slf4j-api/2.0.9"
)
for gav in "${gavs[@]}"; do
  a=$(echo "$gav" | rev | cut -d/ -f2 | rev)
  v=$(echo "$gav" | rev | cut -d/ -f1 | rev)
  curl -sfL -o "$a-$v.jar" "$base/$gav/$a-$v.jar"
  echo "fetched $a-$v.jar"
done
