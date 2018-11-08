package clasDC;

import java.util.List;

import org.datavec.image.data.Image;
import org.nd4j.linalg.api.ndarray.INDArray;

import faults.Fault;
import lombok.Getter;
import utils.FaultUtils;

@Getter
public class CLASFactoryImpl implements CLASFactory {

	private String type;
	private int region;
	private int sector;
	private int nchannels;
	private CLASFactory clasObject = null;

	public CLASFactoryImpl(String type, int nchannels) {
		this(type, nchannels, 1);
	}

	public CLASFactoryImpl(String type, int nchannels, int region) {
		this(type, nchannels, region, 1);
	}

	public CLASFactoryImpl(String type, int nchannels, int region, int sector) {
		this.type = type;
		this.nchannels = nchannels;
		this.region = region;
		this.sector = sector;
		setObject();
	}

	public void setObject() {
		if (type.equalsIgnoreCase("clasdc")) {
			this.clasObject = new CLASDriftChamber(region, sector, nchannels);
		} else if (type.equalsIgnoreCase("clasRegion")) {
			this.clasObject = new CLASDCRegion(region, nchannels);
		} else if (type.equalsIgnoreCase("clas")) {
			this.clasObject = new CLASDCSystem(nchannels);
		}
	}

	public Image getImage() {
		return this.clasObject.getImage();
	}

	public List<Fault> getFaultList() {
		return this.clasObject.getFaultList();

	}

	public static void main(String[] args) {
		CLASFactory factory = new CLASFactoryImpl("clasRegion", 1);
		// INDArray ret = factory.getObject().getImage().getImage();
		INDArray ret = factory.getImage().getImage();

		FaultUtils.draw(factory.getImage());
		int rank = ret.rank();
		int rows = ret.size(rank == 3 ? 1 : 2);
		int cols = ret.size(rank == 3 ? 2 : 3);
		int nchannels = ret.size(rank == 3 ? 0 : 1);
		System.out.println(rank + " " + rows + " " + cols + " " + nchannels);

		for (Fault fault : factory.getFaultList()) {
			fault.printWireInformation();
		}
	}

}
