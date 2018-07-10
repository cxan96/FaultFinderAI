package faultrecordreader;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.writable.Writable;

public class FaultBatchRecordReader extends FaultRecordReader {

	private int currIdx;
	private List<Writable> recordAllocation = new ArrayList<>();
	private FaultWritableRecordBatch currentBatch;

	@Override
	public List<List<Writable>> next(int num) {
		if (currentBatch == null || currIdx >= currentBatch.size()) {
			loadNextBatch();
		}

		if (num == currentBatch.getRecordBatch().size()) {
			currIdx += num;
			return currentBatch;
		} else {
			List<List<Writable>> ret = new ArrayList<>(num);
			int numBatches = 0;
			while (hasNext() && numBatches < num) {
				ret.add(next());
			}

			return ret;
		}

	}

	private void loadNextBatch() {
		// TODO Auto-generated method stub

	}

	public FaultWritableRecordBatch getCurrentBatch() {
		return currentBatch;
	}

}
