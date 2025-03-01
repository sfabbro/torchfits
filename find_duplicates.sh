#!/bin/bash
nm -g build/temp.*/src/torchfits/*.o | grep -v " U " | grep " T " | \
  awk '{print $3}' | sort | uniq -c | sort -nr | grep -v "^      1 "
