#!/bin/bash

# Kafka topics
KAFKA_TOPICS=("metro_air_compressor_data")

KAFKA_SERVER="kafka:9092"

# Wait for Kafka to be ready
until kafka-topics.sh --bootstrap-server "$KAFKA_SERVER" --list; do
  echo "Waiting for Kafka to be ready..."
  sleep 5
done

for TOPIC in "${KAFKA_TOPICS[@]}"; do
    kafka-topics.sh --create --topic "$TOPIC" --bootstrap-server "$KAFKA_SERVER" --partitions 3 --replication-factor 1 || true
done