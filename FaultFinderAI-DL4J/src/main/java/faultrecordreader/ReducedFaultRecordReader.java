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
	factory.getFault(ThreadLocalRandom.current().nextInt(7));
	List<Writable> ret = new ArrayList<>();
	ret.add(new NDArrayWritable(factory.getFeatureVector()));
	ret.add(new IntWritable(getLabelInt(factory.getReducedLabel())));
	return ret;
    }
}
