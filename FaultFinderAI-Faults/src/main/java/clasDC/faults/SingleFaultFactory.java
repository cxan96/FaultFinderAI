package clasDC.faults;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.zip.DataFormatException;

import lombok.Builder;

public class SingleFaultFactory extends AbstractFaultFactory {

	@Builder
	public SingleFaultFactory(int superLayer, int maxFaults, List<FaultNames> desiredFault, boolean randomSuperlayer,
			boolean randomSmear, int nChannels, boolean singleFaultGen) {
		this.superLayer = superLayer;
		this.maxFaults = maxFaults;
		this.desiredFaults = desiredFault;
		this.randomSuperlayer = randomSuperlayer;
		this.randomSmear = randomSmear;
		this.nChannels = nChannels;
		this.singleFaultGen = singleFaultGen;
		initialize();

		loadData();
		generateFaults();
		makeDataSet();
		/**
		 * here I am converting the data set back to x = columns = wires y =
		 * rows = layers
		 */
		convertDataset();

	}

	protected Fault getFault() {
		if (singleFaultGen) {
			return this.getFault(desiredFaults.get(0));
		}
		return this.getFault(desiredFaults.get(ThreadLocalRandom.current().nextInt(0, desiredFaults.size())));
	}

	public int[] getFaultLabel() {
		int[] label = new int[2];
		// lets see if the desired fault is located in the list, if it is, we
		// have the label
		// [1,0]
		// If not the label is
		// [0,1]
		if (faultList.size() == 0) {
			label = IntStream.of(0, 1).toArray();
		} else {
			boolean wantedFound = false;
			for (Fault fault : faultList) {
				if (fault.getSubFaultName().equals(this.desiredFault)) {
					wantedFound = true;
				}
			}
			if (wantedFound) {
				label = IntStream.of(1, 0).toArray();
			} else {
				label = IntStream.of(0, 1).toArray();
			}
		}
		return label;

	}

	public int getSuperLayer() {
		return this.superLayer;
	}

	public List<Fault> getFaultList() {
		return this.faultList;
	}

	// public int getNFaults() {
	// return this.nFaults;
	// }

	public int getLabelInt() {
		getLabelInt(getFaultLabel());
		return this.labelInt;
	}

	public int getLabelInt(int[] labels) {
		for (int i = 0; i < labels.length; i++) {
			if (labels[i] == 1) {
				this.labelInt = i;
				return i;
			}
		}
		this.labelInt = 0;
		return 0;
	}

	public AbstractFaultFactory getNewFactory() {
		AbstractFaultFactory sl = SingleFaultFactory.builder().superLayer(this.superLayer).maxFaults(this.maxFaults)
				.desiredFault(this.desiredFaults).randomSmear(this.randomSmear).nChannels(this.nChannels)
				.singleFaultGen(this.singleFaultGen).build();
		return sl;
	}

	public AbstractFaultFactory getNewFactory(int superLayer) {
		AbstractFaultFactory sl = SingleFaultFactory.builder().superLayer(superLayer).maxFaults(this.maxFaults)
				.desiredFault(this.desiredFaults).randomSmear(this.randomSmear).nChannels(this.nChannels)
				.singleFaultGen(this.singleFaultGen).build();
		return sl;
	}

	public static void main(String[] args) throws DataFormatException {
		AbstractFaultFactory sl = SingleFaultFactory.builder().superLayer(3).maxFaults(1)
				.desiredFault(Stream.of(FaultNames.CONNECTOR_TREE, FaultNames.CHANNEL_THREE)
						.collect(Collectors.toCollection(ArrayList::new)))
				.randomSmear(true).nChannels(1).singleFaultGen(false).build();
		sl.draw();
		System.out.println(sl.getNFaults());

		sl = sl.getNewFactory();
		System.out.println(sl.getNFaults());

		sl.draw();
		sl = sl.getNewFactory(6);
		sl.draw();
		// for (Fault fault : sl.getFaultList()) {
		// fault.printWireInformation();
		// }
		System.out.println(sl.getNFaults());
		// for (FaultNames fault : sl.getDesiredFaults()) {
		// System.out.println(fault.getSaveName());
		// }

	}
}// end
	// of
	// FaultFactory
	// class.
