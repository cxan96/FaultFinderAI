package faultrecordreader;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.AbstractWritableRecordBatch;

public class FaultWritableRecordBatch extends AbstractWritableRecordBatch {

	@Override
	public int size() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public List<Writable> get(int index) {
		// TODO Auto-generated method stub
		return null;
	}

	public List<Writable> getRecordBatch() {
		List<Writable> aList = new ArrayList<>();
		return aList;
	}

}
