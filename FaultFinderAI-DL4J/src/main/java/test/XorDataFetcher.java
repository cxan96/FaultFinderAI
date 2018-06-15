package test;

import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This class is for testing how to set up a custom data fetcher.
 * The data will consist of examples of the XOR-Problem.
 */
public class XorDataFetcher extends BaseDataFetcher {

    public XorDataFetcher() {
        // there are 4 examples in XOR: (0, 0), (1, 0), (0, 1), (1, 1)
        super.totalExamples = 4;
        super.cursor = 0;
        // not entirely sure, but I guess this will be the number of features
        super.inputColumns = 2;
        // this is the number of classes
        super.numOutcomes = 2;
    }

    /**
     * This method basically creates the dataset and stores it in the curr
     * variable of the BaseDataFetcher class.
     */
    @Override
    public void fetch(int numExamples) {
        // create the input data of the XOR problem
        INDArray input = Nd4j.zeros(4, 2);
        input.putScalar(new int [] {0, 0}, 0);
        input.putScalar(new int [] {0, 1}, 0);
        input.putScalar(new int [] {1, 0}, 1);
        input.putScalar(new int [] {1, 1}, 0);
        input.putScalar(new int [] {2, 0}, 0);
        input.putScalar(new int [] {2, 1}, 1);
        input.putScalar(new int [] {3, 0}, 1);
        input.putScalar(new int [] {3, 1}, 1);

        // create the labels of the XOR problem
        // class 0: (1 0), class 1: (0, 1)
        INDArray labels = Nd4j.zeros(4, 2);
        labels.putScalar(new int[] {0, 0}, 1);
        labels.putScalar(new int[] {0, 1}, 0);
        labels.putScalar(new int[] {1, 0}, 0);
        labels.putScalar(new int[] {1, 1}, 1);
        labels.putScalar(new int[] {2, 0}, 0);
        labels.putScalar(new int[] {2, 1}, 1);
        labels.putScalar(new int[] {3, 0}, 1);
        labels.putScalar(new int[] {3, 1}, 0);

        // create the dataset from the input and the labels
        DataSet dataset = new DataSet(input, labels);

        // set the dataset in the parent class
        super.curr = dataset;

	// increment the cursor by 4 examples
	// this is very important, because otherwise it will end up in an infinite loop
	super.cursor+=4;
    }

    /**
     * Not entirely sure if it is necessary to override this one.
     */
    @Override
    public void reset() {
        super.cursor = 0;
        super.curr = null;
    }

    /**
     * Also not sure if overriding this is necessary.
     */
    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }
}
