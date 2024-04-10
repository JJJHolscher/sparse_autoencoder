#! /bin/sh

if [ "$1" = "post-render" ]; then
    find -type f -name '.gitignore' -delete
    cd ../web
    find -type f -name '.gitignore' -delete
fi
