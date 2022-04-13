package org.apache.spark.shuffle.sort;

import org.apache.spark.storage.TempShuffleBlockId;

import java.io.File;

public class RapidsSpillInfo {
    public int numPartitions;
    public File file;
    public TempShuffleBlockId shuffleBlockId;
    public long[] partitionLengths = new long[numPartitions];

    RapidsSpillInfo(int numPartitions, File file, TempShuffleBlockId shuffleBlockId) {
        this.numPartitions =  numPartitions;
        this.file = file;
        this.shuffleBlockId = shuffleBlockId;
    }
}

