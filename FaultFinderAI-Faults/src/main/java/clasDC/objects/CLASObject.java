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

import clasDC.faults.FaultNames;
import lombok.Getter;
import utils.FaultUtils;

/**
 * @author m.c.kunkel
 *
 */
@Getter
public abstract class CLASObject {

	protected String objectType;
	protected int height;
	protected final int width = 112;

	protected int nchannels;
	protected int maxFaults;
	protected List<FaultNames> desiredFaults = null;
	protected boolean singleFaultGen;

	protected double[][] priors;

	protected void setPriors() {
		this.priors = getFixedPriors();
	}

	private double[][] getFixedPriors() {
		if (singleFaultGen) {
			return this.desiredFaults.get(0).getPrior();
		} else {
			List<FaultNames> listDistinct = this.desiredFaults.stream().distinct().collect(Collectors.toList());
			double[][] ret;

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
			return ret;
		}
	}

	public static void main(String[] args) {
		CLASObject clasObject = DriftChamber.builder().region(1).nchannels(1).maxFaults(3).desiredFaults(Stream
				.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_ONE).collect(Collectors.toCollection(ArrayList::new)))
				.singleFaultGen(false).build();
		System.out.println(Arrays.deepToString(clasObject.getPriors()) + "  " + clasObject.getDesiredFaults().size());
	}

}
