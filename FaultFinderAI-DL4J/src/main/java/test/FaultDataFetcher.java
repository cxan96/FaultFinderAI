package test;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;

import faultTypes.FaultData;
import faultTypes.FaultFactory;

public class FaultDataFetcher extends BaseDataFetcher {
	public static final int NUM_EXAMPLES = Integer.MAX_VALUE;
	private FaultFactory factory = null;

	public FaultDataFetcher() {
		totalExamples = NUM_EXAMPLES;
		this.factory = new FaultFactory();
	}

	@Override
	public void fetch(int batch) {
		DataSet dataset = new DataSet();
		for (int i = 0; i < batch; i++) {
			FaultData fData = factory.getFault(1);
			DataSet temp = new DataSet(factory.getFeatureVector(), factory.getLabelVector());
			dataset.addRow(temp, i);
		}
		super.curr = dataset;
	}

	@Override
	public DataSet next() {
		DataSet next = super.next();
		return next;
	}
}
