package faultrecordreader;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class FaultRecorderScaler implements DataNormalization {
	private double minRange;// = FaultUtils.FAULT_RANGE_MIN;
	private double maxRange;// = FaultUtils.RANGE_MAX;

	public FaultRecorderScaler() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void fit(DataSet dataSet) {
		// TODO Auto-generated method stub

	}

	@Override
	public void fit(DataSetIterator iterator) {
		// TODO Auto-generated method stub

	}

	@Override
	public void preProcess(DataSet toPreProcess) {
		INDArray features = toPreProcess.getFeatures();
		this.preProcess(features);
	}

	public void preProcess(INDArray features) {
		this.minRange = (double) features.minNumber();
		this.maxRange = (double) features.maxNumber();
		if (minRange != 0)
			features.subi(minRange); // Offset by minRange
		features.divi((this.maxRange - this.minRange)); // Scaled to 0->1
		// if (this.maxRange - this.minRange != 1)
		// features.muli(this.maxRange - this.minRange); // Scaled to minRange
		// // -> maxRange

	}

	@Override
	public void transform(DataSet toPreProcess) {
		this.preProcess(toPreProcess);
	}

	@Override
	public void transform(INDArray features) {
		this.preProcess(features);
	}

	@Override
	public void transform(INDArray features, INDArray featuresMask) {
		transform(features);

	}

	@Override
	public void transformLabel(INDArray labels) {
		// TODO Auto-generated method stub

	}

	@Override
	public void transformLabel(INDArray labels, INDArray labelsMask) {
		transformLabel(labels);

	}

	@Override
	public void revert(DataSet toRevert) {
		revertFeatures(toRevert.getFeatures());

	}

	@Override
	public NormalizerType getType() {
		return NormalizerType.CUSTOM;

	}

	@Override
	public void revertFeatures(INDArray features) {
		if (minRange != 0) {
			features.subi(minRange);
		}

		features.muli(maxRange - minRange);
	}

	@Override
	public void revertFeatures(INDArray features, INDArray featuresMask) {
		revertFeatures(features);

	}

	@Override
	public void revertLabels(INDArray labels) {
		// TODO Auto-generated method stub

	}

	@Override
	public void revertLabels(INDArray labels, INDArray labelsMask) {
		// TODO Auto-generated method stub

	}

	@Override
	public void fitLabel(boolean fitLabels) {
		if (fitLabels) {
			log.warn(
					"Labels fitting not currently supported for ImagePreProcessingScaler. Labels will not be modified");
		}
	}

	@Override
	public boolean isFitLabel() {
		// TODO Auto-generated method stub
		return false;
	}

}
