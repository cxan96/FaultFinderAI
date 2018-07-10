package faultrecordreader;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

public class FaultRecordReaderDataSetIterator extends RecordReaderDataSetIterator {

	protected FaultRecordReaderDataSetIterator(Builder b) {
		super(b);
		// TODO Auto-generated constructor stub
	}

	@Override
	public boolean hasNext() {
		return (((sequenceIter != null && sequenceIter.hasNext()) || recordReader.hasNext())
				&& (maxNumBatches < 0 || batchNum < maxNumBatches));
	}

}
