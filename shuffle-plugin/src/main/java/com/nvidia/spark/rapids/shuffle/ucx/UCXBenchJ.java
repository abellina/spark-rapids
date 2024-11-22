package com.nvidia.spark.rapids.shuffle.ucx;

import com.nvidia.spark.rapids.ShimLoader$;

public class UCXBenchJ {
    public static void main(String[] args) {
        String configPath = args[0];
        boolean isServer = args[1].equals("-s");
        int numIter = Integer.parseInt(args[2]);
        String localHost = args[3];
        String localPort = args[4];
        String peerHost = isServer ? null : args[5];
        String peerPort = isServer ? null : args[6];
        int maxInFlight = isServer ? 0 : Integer.parseInt(args[7]);
        UCXBench instance =
            (UCXBench) ShimLoader$.MODULE$.newUCXShuffleBench(
                configPath,
                localHost,
                localPort,
                peerHost,
                peerPort,
                maxInFlight,
                numIter);
        instance.start();
    }
}
