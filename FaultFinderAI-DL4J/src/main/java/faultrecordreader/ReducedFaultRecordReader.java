package faultrecordreader;

import org.datavec.api.writable.Writable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The same as FaultRecordReader but uses the reduced labels.
 */
public class ReducedFaultRecordReader extends FaultRecordReader {

    /**
     * Uses the reduced labels.
     */
    @Override
    public List<Writable> next() {
	// label 4 causes problems so generate random faults not equal to 4
	int okLabels [] = {0, 1, 2, 3, 5, 6};
	int curFault = okLabels[ThreadLocalRandom.current().nextInt(6)];
	factory.getFault(curFault);
	List<Writable> ret = new ArrayList<>();
	ret.add(new NDArrayWritable(factory.getFeatureVector()));
	ret.add(new IntWritable(getLabelInt(factory.getReducedLabel())));
	return ret;
    }
}
