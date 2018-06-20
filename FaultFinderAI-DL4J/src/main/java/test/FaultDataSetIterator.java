package test;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

public class FaultDataSetIterator extends BaseDatasetIterator {

	public FaultDataSetIterator(int batch, int numExamples) {
		super(batch, numExamples, new FaultDataFetcher());
	}

}
