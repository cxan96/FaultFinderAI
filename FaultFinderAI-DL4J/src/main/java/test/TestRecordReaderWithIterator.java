/**
 * 
 */
package test;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.datavec.api.records.reader.RecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import clasDC.faults.AbstractFaultFactory;
import clasDC.faults.FaultNames;
import clasDC.faults.SingleFaultFactory;
import faultrecordreader.FaultRecorderScaler;
import faultrecordreader.SingleFaultObjectImageRecordReader;
import lombok.NonNull;
import strategies.FaultRecordScalerStrategy;
import strategies.MinMaxStrategy;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
public class TestRecordReaderWithIterator {
	public static void main(String[] args) {

		int batchSize = 2;
		int batchNum = 2;

		AbstractFaultFactory faultFactory = SingleFaultFactory.builder().superLayer(3).maxFaults(2)
				.desiredFault(Stream.of(FaultNames.CHANNEL_ONE).collect(Collectors.toCollection(ArrayList::new)))
				.randomSmear(true).nChannels(1).singleFaultGen(true).build();

		RecordReader recordReader = new SingleFaultObjectImageRecordReader(faultFactory, 5, 111, 6, 112);
		FaultRecordScalerStrategy strategy = new MinMaxStrategy();

		DataSetIterator test = new RecordReaderDataSetIterator.Builder(recordReader, batchSize).regression(1)
				.maxNumBatches(batchNum).preProcessor(new FaultRecorderScaler(strategy)).build();
		//
		// for (int i = 0; i < test.batch(); i++) {
		// DataSet ds = test.next();
		// // doStuff(ds);
		// }
		int numEpochs = 2;
		for (int i = 0; i < numEpochs; i++) {
			System.out.println("EPOCH " + i);
			MultiDataSet mds = makeMuliDataSet(test);
			fit(mds.getFeatures(), mds.getLabels(), mds.getFeaturesMaskArrays(), mds.getLabelsMaskArrays());

		}

	}

	private static void doStuff(DataSet ds) {
		System.out.println(ds.numInputs() + "  " + ds.getFeatures().shapeInfoToString());
		for (int i = 0; i < ds.numExamples(); i++) {
			INDArray out = ds.getFeatures().slice(i);
			System.out.println(out.shapeInfoToString());
			for (int j = 0; j < out.size(1); j++) {
				System.out.println(ds.getFeatures().getDouble(i, 0, j, 0));

			}

			FaultUtils.draw(out);
		}
	}

	public static MultiDataSet makeMuliDataSet(@NonNull DataSetIterator iterator) {
		MultiDataSetIterator ret = new MultiDataSetIteratorAdapter(iterator);
		if (!ret.hasNext() && ret.resetSupported()) {
			ret.reset();
		}
		MultiDataSet mds = null;
		while (ret.hasNext()) {
			mds = ret.next();
		}
		return mds;
	}

	public static void fit(@NonNull DataSetIterator iterator, int numEpochs) {
		Preconditions.checkArgument(numEpochs > 0, "Number of epochs much be > 0. Got numEpochs = %s", numEpochs);
		Preconditions.checkArgument(numEpochs == 1 || iterator.resetSupported(),
				"Cannot perform multiple epochs training using"
						+ "iterator thas does not support resetting (iterator.resetSupported() returned false)");

		for (int i = 0; i < numEpochs; i++) {
			System.out.println("EPOCH " + i);
			fit(iterator);
		}
	}

	public static void fit(@NonNull DataSetIterator iterator) {
		fit(new MultiDataSetIteratorAdapter(iterator));
	}

	public static void fit(MultiDataSetIterator multi) {

		if (!multi.hasNext() && multi.resetSupported()) {
			multi.reset();
		}

		boolean destructable = false;

		MultiDataSetIterator multiDataSetIterator;

		multiDataSetIterator = multi;

		while (multiDataSetIterator.hasNext()) {
			MultiDataSet mds = multiDataSetIterator.next();
			fit(mds.getFeatures(), mds.getLabels(), mds.getFeaturesMaskArrays(), mds.getLabelsMaskArrays());
		}
	}

	public static void fit(INDArray[] inputs, INDArray[] labels, INDArray[] featureMaskArrays,
			INDArray[] labelMaskArrays) {
		System.out.println(inputs.length + "   lenght of arrays   ");
		for (INDArray in : inputs) {
			for (int i = 0; i < in.size(0); i++) {
				FaultUtils.draw(in.slice(i));
			}
			System.out.println(in.rank() + "   rank!!!!!!!    " + in.shapeInfoToString());
		}

	}

}
