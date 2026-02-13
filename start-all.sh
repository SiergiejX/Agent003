#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "▶ Sprawdzanie Dockera..."
command -v docker >/dev/null || { echo "❌ Docker nie jest zainstalowany"; exit 1; }
docker compose version >/dev/null || { echo "❌ Docker Compose niedostępny"; exit 1; }

start_compose () {
  local dir="$1"
  echo
  echo "▶ Uruchamianie: $dir"
  cd "$BASE_DIR/$dir"
  docker compose up -d
}

echo
echo "=== START SYSTEMU ==="

# Kontenery bazowe
start_compose "ollama"

# Czekaj aż Ollama będzie gotowa i pobierz model phi3:mini
echo
echo "▶ Czekanie na gotowość Ollama..."
sleep 5
for i in {1..12}; do
  if docker exec ollama ollama list &>/dev/null; then
    echo "✅ Ollama gotowa"
    break
  fi
  echo "   Próba $i/12..."
  sleep 5
done

echo "▶ Pobieranie modelu phi3:mini..."
docker exec ollama ollama pull phi3:mini
echo "✅ Model phi3:mini zainstalowany"
echo "▶ Pobieranie modelu embendingowego..."
docker exec ollama pull nomic-embed-text
echo "✅ Model embendingowy zainstalowany"

start_compose "qdrant"
start_compose "Open_WebUI"
start_compose "nodered"

# Agenci
start_compose "agents/agent1_student"
start_compose "agents/agent2_ticket"
start_compose "agents/agent3_analytics"
start_compose "agents/agent4_bos"
start_compose "agents/agent5_security"

echo
echo "✅ Wszystkie kontenery zostały uruchomione"
echo
docker ps
