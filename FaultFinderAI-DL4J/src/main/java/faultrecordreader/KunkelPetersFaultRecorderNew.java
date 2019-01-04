package faultrecordreader;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;

import clasDC.factories.FaultFactory;
import clasDC.faults.Fault;
import clasDC.faults.FaultNames;
import clasDC.objects.CLASObject;
import clasDC.objects.CLASObject.ContainerType;
import clasDC.objects.SuperLayer;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;
import utils.FaultUtils;

public class KunkelPetersFaultRecorderNew implements RecordReader {

	protected FaultFactory factory = null;
	private int label;

	public KunkelPetersFaultRecorderNew(int superLayer, int maxFaults, FaultNames desiredFault,
			boolean singleFaultGeneration, boolean blurredFaults, int nchannels) {

		this(SuperLayer.builder().superlayer(superLayer).nchannels(nchannels).minFaults(1).maxFaults(maxFaults)
				.desiredFault(desiredFault).singleFaultGen(singleFaultGeneration).containerType(ContainerType.OBJ)
				.build());
	}

	public KunkelPetersFaultRecorderNew(CLASObject clasObject) {
		if (clasObject instanceof SuperLayer) {
			SuperLayer obj = (SuperLayer) clasObject;
			this.factory = new FaultFactory(obj.getSuperlayer(), obj.getMaxFaults(), obj.getDesiredFault(),
					obj.isSingleFaultGen(), true, obj.getNchannels());
		} else {
			throw new IllegalArgumentException("Only superlayer object is valid");
		}

	}

	@Override
	public boolean batchesSupported() {
		// I might want to set this to true so that I train in batches, reduces
		// memory
		// will get back to this after impl of modes
		return true;
	}

	@Override
	public boolean hasNext() {
		// since this is batch mode, and the data is generated on the fly, this
		// should always be true
		return true;
	}

	@Override
	public List<Writable> next() {
		List<Writable> ret = new ArrayList<>();

		ret.add(new NDArrayWritable(factory.asImageMatrix().getImage()));
		ret.add(new IntWritable(getLabelInt(factory.getFaultLabel())));

		this.reset();
		return ret;
	}

	@Override
	public void reset() {
		this.factory = this.factory.getNewFactory();
	}

	@Override
	public boolean resetSupported() {
		// Why would we need to reset in this type of training?
		return true;
	}

	@Override
	public List<List<Writable>> next(int batch) {
		List<List<Writable>> ret = new ArrayList<>();
		for (int i = 0; i < batch && hasNext(); i++) {
			ret.add(next());
		}
		return ret;
	}

	@Override
	public void close() throws IOException {
		// TODO Auto-generated method stub

	}

	@Override
	public Configuration getConf() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setConf(Configuration arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<String> getLabels() {
		List<String> labels = new ArrayList<>();
		for (Fault faults : factory.getFaultList()) {
			labels.add(faults.getFaultName());
		}
		return null;
	}

	@Override
	public List<RecordListener> getListeners() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initialize(InputSplit arg0) throws IOException, InterruptedException {
		// TODO Auto-generated method stub

	}

	@Override
	public void initialize(Configuration arg0, InputSplit arg1) throws IOException, InterruptedException {
		// TODO Auto-generated method stub

	}

	@Override
	public Record loadFromMetaData(RecordMetaData arg0) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<Record> loadFromMetaData(List<RecordMetaData> arg0) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Record nextRecord() {
		List<Writable> list = next();
		URI uri = URI.create("FaultFinder");
		// return new org.datavec.api.records.impl.Record(list, metaData)
		return new org.datavec.api.records.impl.Record(list,
				new RecordMetaDataURI(null, KunkelPetersFaultRecorderNew.class));

	}

	@Override
	public List<Writable> record(URI arg0, DataInputStream arg1) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setListeners(RecordListener... arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setListeners(Collection<RecordListener> arg0) {
		// TODO Auto-generated method stub

	}

	protected int getLabelInt(int[] labels) {
		for (int i = 0; i < labels.length; i++) {
			if (labels[i] == 1) {
				this.label = i;
				return i;
			}
		}
		this.label = 0;
		return 0;
	}

	public int getLabelInt() {
		return this.label;
	}

	public void initialize() {

	}

	private void test(int i) {

		FaultUtils.draw(factory.asImageMatrix().getImage(), " Plot " + i);

		System.out.println(getLabelInt(factory.getFaultLabel()) + "   for " + i);

		this.reset();

	}

	public static void main(String[] args) throws IOException {

		CLASObject clasObject = SuperLayer.builder().superlayer(1).nchannels(1).minFaults(1).maxFaults(7)
				.desiredFault(FaultNames.CHANNEL_ONE)
				.desiredFaults(Stream.of(FaultNames.FUSE_A, FaultNames.CHANNEL_ONE)
						.collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).isScaled(false).containerType(ContainerType.OBJ).build();

		FaultRecordScalerStrategy strategy = new MinMaxStrategy();
		KunkelPetersFaultRecorderNew recordReader = new KunkelPetersFaultRecorderNew(2, 20, FaultNames.PIN_BIG, true, true,
				1);
		// KunkelPetersFaultRecorder recordReader = new
		// KunkelPetersFaultRecorder(clasObject);
		for (int i = 0; i < 10; i++) {
			recordReader.test(i);

		}
		recordReader.close();
	}
}
