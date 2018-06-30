package test;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;

import client.FaultClassifier;
import client.ModelFactory;
import faultrecordreader.ReducedFaultRecordReader;

import java.util.Scanner;

public class FaultClassifierTest {
    public static void main (String args[]) {
        // get the desired model
        MultiLayerNetwork model = ModelFactory.simpleCNN(13);

        // initialize the fault classifier
        FaultClassifier classifier = new FaultClassifier(model);

        // set up a local web-UI to monitor the training available at localhost:9000
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        // additionally print the score on every iteration
        classifier.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));
        uiServer.attach(statsStorage);

        // train the classifier
        classifier.train(20, 5000, new ReducedFaultRecordReader());

        // evaluate the classifier
        String stats = classifier.evaluate(1, 5000, new ReducedFaultRecordReader());
        System.out.println(stats);

        // press enter to exit the program
        // this will tear down the web ui
        Scanner sc = new Scanner(System.in);
        System.out.println("Press enter to exit.");
        sc.nextLine();

        System.exit(0);
    }
}
