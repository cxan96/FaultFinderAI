package test;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import faultrecordreader.FaultRecordReader;
import faultrecordreader.FaultRecordReaderDataSetIterator;
import faultrecordreader.FaultRecorderScaler;
import faultrecordreader.ReducedFaultRecordReader;

public class TestNewNormalization {

	public static void main(String[] args) {
		FaultRecordReader recordReader = new ReducedFaultRecordReader();
		// List<Writable> aList = recordReader.next();
		// System.out.println(aList.get(0).toString());

		// DataSetIterator dataIter = new
		// RecordReaderDataSetIterator(recordReader, 10, 1, 14, 2);
		int batchNum = 2;
		int batchSize = 10;
		DataSetIterator dataIter = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
				// currently there are 14 labels in the dataset
				.classification(1, 14).maxNumBatches(batchNum).preProcessor(new FaultRecorderScaler()).build();

		DataSetIterator testIter = new FaultRecordReaderDataSetIterator.Builder(recordReader, batchSize)
				// currently there are 14 labels in the dataset
				.classification(1, 14).maxNumBatches(batchNum).preProcessor(new FaultRecorderScaler()).build();
		// DataSet dSet = dataIter.next(1);
		// System.out.println(dSet.getFeatures().toString());
		System.out.println(dataIter.batch() + "  numExamples");

		int counter = 0;

		// pretend numEpochs = N = 3

		int numEpochs = 3;
		for (int i = 0; i < numEpochs; i++) {
			while (dataIter.hasNext()) {
				DataSet dSet = dataIter.next();
				System.out.println(dSet.getFeatures().toString());
				counter++;
				System.out.println("counter = ##################" + counter + " ##################### epoc =  " + i);

			}
		}

		// dataIter.reset();
		//
		// while (dataIter.hasNext()) {
		// DataSet dSet = dataIter.next();
		// System.out.println(dSet.getFeatures().toString());
		// counter++;
		// System.out.println("MORE counter = ##################" + counter + "
		// #####################");
		//
		// }

		// DataNormalization scaler = new FaultRecorderScaler();
		// scaler.fit(dataIter);
		// dataIter.setPreProcessor(scaler);
		// dSet = dataIter.next(1);
		// System.out.println("########### Features ################");
		// System.out.println(dSet.getFeatures().toString());
		//
		// double max = (double) dSet.getFeatures().maxNumber();
		// double min = (double) dSet.getFeatures().minNumber();
		//
		// H1F aH1f = new H1F("adub", 100, min, max);
		// for (int i = 0; i < dSet.getFeatures().length(); i++) {
		// double adub = dSet.getFeatures().getDouble(i);
		//
		// aH1f.fill(adub);
		// }
		// TCanvas canvas = new TCanvas("Canvas", 800, 800);
		// canvas.draw(aH1f);

	}
}
