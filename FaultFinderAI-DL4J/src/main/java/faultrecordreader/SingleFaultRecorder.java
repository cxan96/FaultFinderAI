package faultrecordreader;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;

public class SingleFaultRecorder extends FaultRecordReader {

	private int faultType;

	public SingleFaultRecorder(int faultType) {
		this.faultType = faultType;
	}

	@Override
	public List<Writable> next() {
		factory.getFault(faultType);
		List<Writable> ret = new ArrayList<>();
		ret.add(new NDArrayWritable(factory.getFeatureVector()));
		ret.add(new IntWritable(getLabelInt(factory.getFaultLabel())));
		// System.out.println(Arrays.toString(factory.getFeatureArray()));
		// System.out.println(Arrays.toString(factory.getFaultLabel()));
		// factory.plotData();

		return ret;
	}
}
