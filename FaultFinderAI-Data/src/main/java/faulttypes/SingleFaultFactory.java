package faulttypes;

import java.util.concurrent.ThreadLocalRandom;

import org.jlab.groot.data.H1F;
import org.jlab.groot.ui.TCanvas;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.NDArrayUtil;

public class SingleFaultFactory extends FaultFactory {
	private int[] label = null;
	private int[] reducedLabel = null;
	private int[][] featureData = null;
	private String type = null;
	private FaultData retFault = null;
	private boolean withHotwireFault;
	private String faultName;

	public SingleFaultFactory() {
		this(true);
	}

	public SingleFaultFactory(boolean withHotwireFault) {
		this.withHotwireFault = withHotwireFault;

	}

	// use getFault method to get object of type Plan
	@Override
	public FaultData getFault(int type) {
		int startValue = super.getFault(type).getReducedLabel().length;
		int maxValue = startValue + 5;
		int rndm = ThreadLocalRandom.current().nextInt(maxValue);
		// System.out.println(rndm);

		if (rndm >= maxValue - 1) {
			this.retFault = getAnythingElse(type);
			this.reducedLabel = makeLabel(new int[super.getFault(type).getReducedLabel().length],
					new int[super.getFault(4).getReducedLabel().length], new int[] { 1 });
		} else if (rndm == maxValue - 2) {
			this.retFault = super.getFault(4);
			this.reducedLabel = makeLabel(new int[super.getFault(type).getReducedLabel().length],
					this.retFault.getReducedLabel(), new int[] { 0 });
		} else {

			this.retFault = super.getFault(type);
			this.reducedLabel = makeLabel(this.retFault.getReducedLabel(),
					new int[super.getFault(4).getReducedLabel().length], new int[] { 0 });

		}

		this.featureData = this.retFault.getData();
		setFaultName(super.getFault(type).getClass().getSimpleName());

		// System.out.println(this.retFault.getClass().getSimpleName());
		return this.retFault;
	}

	private FaultData getAnythingElse(int type) {
		int rndm = getRandomNumber(type);
		return super.getFault(rndm);
	}

	@Override
	public void plotData() {
		this.retFault.plotData();
	}

	private int getRandomNumber(int type) {
		int rndm = ThreadLocalRandom.current().nextInt(withHotwireFault ? 7 : 6);
		if (rndm == type || rndm == 4) {
			rndm = getRandomNumber(type);
		}
		return rndm;

	}

	private int[] makeLabel(int[]... coord) {
		int[] aLabel = new int[getArraySize(coord)];
		int sizePlacer = 0;
		for (int i = 0; i < coord.length; i++) {
			System.arraycopy(coord[i], 0, aLabel, sizePlacer, coord[i].length);
			sizePlacer += coord[i].length;
		}
		return aLabel;
	}

	private int getArraySize(int[]... coord) {
		int size = 0;
		for (int[] ints : coord) {
			size += ints.length;
		}
		return size;
	}

	public INDArray getLabelVector() {
		return NDArrayUtil.toNDArray(this.label);
	}

	public INDArray getFeatureVector() {
		return NDArrayUtil.toNDArray(ArrayUtil.flatten(this.featureData));
	}

	public int[] getFeatureArray() {
		return ArrayUtil.flatten(this.featureData);
	}

	public int[] getLabel() {
		return this.label;
	}

	/**
	 * 
	 * getFaultLabel(): The label for an individual fault i.e. HVChannel will
	 * have int[8]
	 * 
	 */

	public int[] getFaultLabel() {
		return this.retFault.getLabel();
	}

	public int[] getReducedLabel() {
		return reducedLabel;
	}

	public int[][] getFeatureData() {
		return featureData;
	}

	public String getType() {
		return type;
	}

	public int getReducedFaultIndex() {
		int retVal = 0;
		for (int i = 0; i < this.reducedLabel.length; i++) {
			if (this.reducedLabel[i] == 1) {
				retVal = i;
			}
		}
		return retVal;
	}

	public String getFaultName() {
		return this.faultName;
	}

	public String getFaultNameII() {
		return this.retFault.getClass().getSimpleName();
	}

	private void setFaultName(String faultName) {
		this.faultName = faultName;
	}

	public static void main(String[] args) {
		for (int j = 0; j < 10; j++) {

			H1F aH1f = new H1F("name", "title", 6, 1, 6);
			int faultType = 0;
			for (int i = 0; i < 3000; i++) {
				SingleFaultFactory sFactory = new SingleFaultFactory();
				sFactory.getFault(faultType);
				if (sFactory.getFaultNameII()
						.equals(new FaultFactory().getFault(faultType).getClass().getSimpleName())) {
					aH1f.fill(1);
				} else if (sFactory.getFaultNameII().equals("HVNoFault")) {
					aH1f.fill(2);
				} else {
					aH1f.fill(3);
				}
				// System.out.println(Arrays.toString(sFactory.getReducedLabel()));
				// sFactory.plotData();
			}
			TCanvas canvas = new TCanvas("name", 800, 1200);
			aH1f.normalize(aH1f.getEntries());
			canvas.draw(aH1f);
		}
	}
}// end
	// of
	// FaultFactory
	// class.
