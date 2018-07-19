package faultrecordreader;

import java.util.ArrayList;
import java.util.List;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;

import faulttypes.FaultNames;
import faulttypes.SingleFaultFactory;

public class SmallPinFaultRecordReader extends FaultRecordReader {

	private SingleFaultFactory singleFactory = new SingleFaultFactory();

	@Override
	public List<Writable> next() {
		// first, generate a pin fault from the factory
		singleFactory.getFault(0);

		// this is the label that we will build
		// the first number indicates small pin, the second not a small pin
		int label = 0;

		// see, what we are dealing with
		if (singleFactory.getFaultNameII().equals("HVPinFault")) {
			// it is a pin, but is it also a small pin?
			if (singleFactory.getReducedLabel() == FaultNames.PIN_SMALL) {
				// small pin
				label = 1;
			} else {
				// not a small pin
				label[1] = 1;
			}
		} else {
			// not a small pin
			label[1] = 1;
		}

		// now that we know the label, lets create the return object
		List<Writable> ret = new ArrayList<>();
		ret.add(new NDArrayWritable(singleFactory.getFeatureVector()));
		ret.add(new IntWritable(getLabelInt(label)));
		return ret;
	}

	public static void main(String args[]) {
		FaultRecordReader reader = new SmallPinFaultRecordReader();
		for (int i = 0; i < 20; i++) {
			System.out.println(reader.next().get(1));
		}
	}
}
