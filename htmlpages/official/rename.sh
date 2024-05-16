for file in *.json; do
    mv "$file" "${file%.json}"
done

