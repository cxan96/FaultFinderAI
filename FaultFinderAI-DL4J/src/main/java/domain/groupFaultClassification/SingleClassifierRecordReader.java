package domain.groupFaultClassification;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;

import faultrecordreader.FaultRecordReader;
import faulttypes.FaultFactory;
import faulttypes.SingleFaultFactory;

/**
 * Purpose: to call the modified FaultFactory method getSingleFault which
 * consists of a singleFault + noFault+ everything else
 */
public class SingleClassifierRecordReader extends FaultRecordReader {

	private int faultType;
	protected FaultFactory factory = null;

	public SingleClassifierRecordReader(int faultType) {
		this.factory = new SingleFaultFactory();
		this.faultType = faultType;
	}

	@Override
	public List<Writable> next() {
		this.factory.getFault(faultType);
		List<Writable> ret = new ArrayList<>();
		ret.add(new NDArrayWritable(this.factory.getFeatureVector()));
		ret.add(new IntWritable(getLabelInt(this.factory.getReducedLabel())));
		// System.out.println(Arrays.toString(this.factory.getReducedLabel()));
		return ret;
	}

}
