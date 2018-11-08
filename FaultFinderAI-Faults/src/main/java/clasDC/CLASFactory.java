package clasDC;

import java.util.List;

import org.datavec.image.data.Image;

import faults.Fault;

public interface CLASFactory {

	Image getImage();

	List<Fault> getFaultList();

}
