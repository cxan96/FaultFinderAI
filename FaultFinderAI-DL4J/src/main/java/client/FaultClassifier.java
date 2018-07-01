package client;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.eval.Evaluation;

import faultrecordreader.FaultRecordReader;
import faultrecordreader.FaultRecorderScaler;

import java.io.File;
import java.io.IOException;

public class FaultClassifier {

    /** The network that is used as the underlying model. */
    private MultiLayerNetwork network;

    /**
     * Initializes the FaultClassifier using a saved model.
     *
     * @param fileName The file that the model is loaded from.
     */
    public FaultClassifier(String fileName) throws IOException{
        this.network = ModelSerializer.restoreMultiLayerNetwork(new File(fileName));
    }

    /**
     * Initializes the FaultClassifier using a readily configured network.
     *
     * @param network The network that serves as the underlying model of FaultClassifier.
     */
    public FaultClassifier(MultiLayerNetwork network) {
        this.network = network;
    }

    /**
     * Trains the underlying model.
     *
     * @param batchSize The size of each batch.
     * @param batchNum The amount of batches to pass through during training.
     * @param recordReader The FaultRecordReader to be used.
     */
    public void train (int batchSize, int batchNum, FaultRecordReader recordReader) {
        // set up the DatasetIterator
        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
	    // currently there are 13 labels in the dataset
            .classification(1, 13)
            .maxNumBatches(batchNum)
            .preProcessor(new FaultRecorderScaler())
            .build();

        // this trains the model on batchNum batches
        this.network.fit(iterator);
    }

    /**
     * Saves the underlying model to a file.
     * The current state of the updater is not saved.
     *
     * @param fileName The file that the model is saved to.
     */
    public void save(String fileName) throws IOException{
        ModelSerializer.writeModel(this.network, new File(fileName), false);
    }

    /**
     * Evaluates the model.
     *
     * @param batchSize The size of each batch.
     * @param batchNum The amount of batches to pass through during evaluation.
     * @param recordReader The FaultRecordReader to be used.
     *
     * @return The results of the Evaluation.
     */
    public String evaluate(int batchSize, int batchNum, FaultRecordReader recordReader) {
        // set up the DatasetIterator
        DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
	    // currently there are 13 labels in the dataset
            .classification(1, 13)
            .maxNumBatches(batchNum)
            .preProcessor(new FaultRecorderScaler())
            .build();

        // perform the testing
        Evaluation evaluation = this.network.evaluate(iterator);

        // return the results
        return evaluation.stats();
    }

    /**
     * Set the listeners for the model. Use this to monitor training.
     */
    public void setListeners(TrainingListener... listeners) {
        this.network.setListeners(listeners);
    }
}
