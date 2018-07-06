package client;

import org.eclipse.jgit.transport.CredentialsProvider;
import org.eclipse.jgit.transport.UsernamePasswordCredentialsProvider;
import org.eclipse.jgit.lib.Ref;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.storage.file.FileRepositoryBuilder;

import java.io.Console;
import java.util.List;

public class GitTrainer {
    
    public static void main (String args[]) throws Exception{
	CredentialsProvider credentials = authenticate();
	verifyCredentials(credentials);
	
    }

    private static CredentialsProvider authenticate() {
	Console console = System.console();
	String userName = console.readLine("Username: ");
	char [] password = console.readPassword("Password: ");

	CredentialsProvider res = new UsernamePasswordCredentialsProvider(userName, password);
	return res;
    }

    private static void verifyCredentials(CredentialsProvider credentials) throws Exception{
	Repository repo = new FileRepositoryBuilder()
	    .setMustExist(true)
	    .findGitDir()
	    .build();

	Git git = new Git(repo);

	// test if a push is possible to verify the credentials
	git.push().setCredentialsProvider(credentials).add("training").call();
    }
}
