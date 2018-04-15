import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

class Rohs {
    public static void main(String[] args) throws IOException, InterruptedException {
        ProcessBuilder pb;
        switch(System.getProperty("os.name")) {
            case "Mac OS X":
                pb = new ProcessBuilder(
                    "/usr/bin/script", "-q", "/dev/null", "python");
                break;
            default:
                // Linux
                pb = new ProcessBuilder(
                    "/usr/bin/script", "-qfc", "python -i init.py", "/dev/null");

        }

        Process p = pb.start();

        char[] readBuffer = new char[1000];
        InputStreamReader isr = new InputStreamReader(p.getInputStream());
        BufferedReader br = new BufferedReader(isr);
        int charCount;
        boolean written = false;
        while(true) {
            if (!br.ready() && !written) {
                // Ugly. Should be reading for '>>>' prompt then writing.
                Thread.sleep(1000);
                if (!written) {
                    written = true;
                    OutputStream os = p.getOutputStream();
                    OutputStreamWriter osw = new OutputStreamWriter(os);
                    BufferedWriter bw = new BufferedWriter(osw);
                    bw.write("im = Image.open(\"input.jpg\")");
                    bw.newLine();
                    bw.write("a = rescale(im)");
                    bw.newLine();
                    bw.write("a = Variable(a)");
                    bw.newLine();
                    bw.write("a = netG(a.view(-1, 3, 128, 128))");
                    bw.newLine();
                    bw.write("vutils.save_image(a.data, 'result.png', normalize=True)");
                    bw.newLine();
                    bw.flush();
                    bw.close();
                }
                continue;
            }
            charCount = br.read(readBuffer);
            if (charCount > 0)
                System.out.print(new String(readBuffer, 0, charCount));
            else
                break;
        }
    }
}
