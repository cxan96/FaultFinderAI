/**
 * 
 */
package clasDC.objects;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.lang3.tuple.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import clasDC.faults.FaultNames;
import clasDC.faults.FaultSlidingInformation;
import lombok.Getter;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
@Getter
public abstract class CLASObject {
	/**
	 * Enum for type of object <br>
	 * OBJ is reserved for object detection (not yet working) <br>
	 * SEG is reserved for image segmentation
	 */
	public enum ContainerType {
		CLASS, MULTICLASS, OBJ, SEG;
	}

	protected ContainerType containerType;

	protected String objectType;
	protected int height;
	protected int width = 112;

	protected int nchannels;
	protected int minFaults;
	protected int maxFaults;

	protected FaultNames desiredFault = null;
	protected double desiredFaultGenRate;
	protected List<FaultNames> desiredFaults = null;
	protected boolean singleFaultGen;
	protected boolean isScaled = false;

	protected double[][] priors;

	protected int nLabels;
	protected int batchSize;

	protected void setPriors() {
		this.priors = getFixedPriors();
		setHeightWith();

	}

	private double[][] getFixedPriors() {
		double[][] ret;

		if (this.desiredFaults != null) {

			List<FaultNames> listDistinct = this.desiredFaults.stream().distinct().collect(Collectors.toList());

			if (listDistinct.contains(FaultNames.NOFAULT)) {
				listDistinct.remove(FaultNames.NOFAULT);
			}
			Set<Pair<Double, Double>> pairPriors = new LinkedHashSet<>();
			for (FaultNames name : listDistinct) {
				pairPriors.add(Pair.of(name.getPrior()[0][0], name.getPrior()[0][1]));
			}
			List<Pair<Double, Double>> aList = new ArrayList<>(pairPriors);
			double[][] temppriors = new double[][] { { aList.get(0).getLeft(), aList.get(0).getRight() } };
			ret = temppriors;

			for (int i = 1; i < aList.size(); i++) {
				temppriors = new double[][] { { aList.get(i).getLeft(), aList.get(i).getRight() } };

				ret = FaultUtils.merge(ret, temppriors);

			}

			if (isScaled) {
				int[] tempscale = FaultUtils.scaleImage(this.height, this.width);
				double[][] scales = { { 1.0 / (double) tempscale[1], 1.0 / (double) tempscale[0] } };
				ret = FaultUtils.getPriors(ret, scales);
			}
		} else {
			ret = new double[][] { { 0, 0 } };
		}
		return ret;

	}

	private void setHeightWith() {
		if (isScaled) {
			int[] tempscale = FaultUtils.scaleImage(this.height, this.width);
			this.height = this.height * tempscale[0] + 2 * tempscale[2];
			this.width = this.width * tempscale[1] + 2 * tempscale[3];
		}
	}

	public int getNLabels() {
		// OK the logic is as follows
		// If we are doing object detection, then we will
		// use the sliding window method, which is either binary: 0 not there; 1
		// there
		// or in the case of Pins and wires, multiclass
		// 1,0,0 there at position 1;
		// 0,1,0 there at position 2
		// 1,1,0 there at position 1 and 2
		// 0,0,1 not there
		// If not object detection, we must be doing classification as a whole
		// not yet impl classification as a whole
		if (this.containerType.equals(ContainerType.OBJ)) {

			FaultSlidingInformation faultSlidingInformation = this.desiredFault.getFSlidingInformation();
			if (faultSlidingInformation.getYInc() == 0) {
				return 2;
			} else {
				return 3;
			}
		} else if (this.containerType.equals(ContainerType.MULTICLASS)) {

			return (int) this.desiredFault.getPossiblePositions().length();

		} else {
			return 2;
		}

	}

	/**
	 * This batchSize is determined by the amount of possible locations the
	 * object can take place
	 * 
	 * @return the batchsize
	 */
	public int getBatchSize() {
		if (this.containerType.equals(ContainerType.OBJ)) {
			if (this.desiredFault.equals(FaultNames.PIN_SMALL) || this.desiredFault.equals(FaultNames.PIN_BIG)) {
				return (int) this.desiredFault.getPossiblePositions().length() / 2;

			} else {
				return (int) this.desiredFault.getPossiblePositions().length();
			}
		} else if (this.containerType.equals(ContainerType.MULTICLASS)) {
			return 1;
		} else {
			return 1;
		}
	}

	public static void main(String[] args) {
		CLASObject clasObject = SuperLayer.builder().superlayer(1).nchannels(1).minFaults(2).maxFaults(3)
				.desiredFault(FaultNames.CHANNEL_ONE)
				.desiredFaults(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_TWO)
						.collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).isScaled(false).containerType(ContainerType.OBJ).desiredFaultGenRate(0.5)
				.build();

		System.out.println(Arrays.deepToString(clasObject.getPriors()) + "  " + clasObject.getDesiredFaults().size()
				+ "  " + clasObject.getMinFaults() + "  " + clasObject.getDesiredFault());

		INDArray test = Nd4j.zeros(6, 10);
		System.out.println(test.length() + "   length");

	}

}
