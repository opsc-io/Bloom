#!/bin/bash
# Wait for CockroachDB and create database
until cockroach sql --insecure --host=db -e "SELECT 1" > /dev/null 2>&1; do
  echo "Waiting for CockroachDB..."
  sleep 2
done
cockroach sql --insecure --host=db -e "CREATE DATABASE IF NOT EXISTS bloom;"
npx prisma migrate deploy
