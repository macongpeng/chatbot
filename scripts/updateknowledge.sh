#!/bin/sh
echo "Updating Medirecords Knowledge Base"

rm -rf htmlpages/knowledge

mkdir -p htmlpages/knowledge
python3 downloadknowledge.py