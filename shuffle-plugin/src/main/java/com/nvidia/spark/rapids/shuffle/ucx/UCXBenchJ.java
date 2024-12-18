package com.nvidia.spark.rapids.shuffle.ucx;

import com.nvidia.spark.rapids.ShimLoader$;

public class UCXBenchJ {
    public static void main(String[] args) {
        String configPath = args[0];
        boolean isServer = args[1].equals("-s");
        int numIter = Integer.parseInt(args[2]);
        String localHost = args[3];
        String localPort = args[4];
        long msgSize = Long.parseLong(args[5]);
        int maxInFlight = Integer.parseInt(args[6]);
        String peerHost = isServer ? null : args[7];
        String peerPort = isServer ? null : args[8];
        UCXBench instance =
            (UCXBench) ShimLoader$.MODULE$.newUCXShuffleBench(
                configPath,
                localHost,
                localPort,
                peerHost,
                peerPort,
                maxInFlight,
                numIter,
                msgSize);
        instance.start();
    }
}
