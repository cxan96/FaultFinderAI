package faultrecordreader;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;

/**
 * The same as FaultRecordReader but uses the reduced labels.
 */
public class ReducedFaultRecordReader extends FaultRecordReader {

	/**
	 * Uses the reduced labels.
	 */
	@Override
	public List<Writable> next() {
	    int faults [] = {0, 1, 2, 3, 4, 5, 6};
		factory.getFault(faults[ThreadLocalRandom.current().nextInt(faults.length)]);
		// if (faultNum == 4) {
		// System.out.println("We did a Nofault label 13");
		// }
		List<Writable> ret = new ArrayList<>();
		ret.add(new NDArrayWritable(factory.getFeatureVector()));
		ret.add(new IntWritable(getLabelInt(factory.getReducedLabel())));
		return ret;
	}
}
