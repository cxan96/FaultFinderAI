package test;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

/**
 * This class is for testing how to set up a custom DatasetIterator.
 * The data will consist of examples of the XOR-Problem.
 * See XorDataFetcher for more details.
 */
public class XorDataSetIterator extends BaseDatasetIterator{

    public XorDataSetIterator() {
	// first parameter: batch (not yet really sure what this does)
	// second parameter: number of examples (there are 4 examples in the XOR data set)
	super(1, 4, new XorDataFetcher());
    }
}
