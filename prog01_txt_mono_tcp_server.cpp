//----------------------------------------------------------------------------

#include "crsUtils.hpp"

int
main(int argc,
     char **argv)
{
std::vector<std::string> args(argv,argv+argc);

//---- check command line arguments ----
if(args.size()!=2)
  {
  crs::write(STDERR_FILENO,crs::txt("usage: % port\n",args[0]));
  crs::exit(1);
  }

//---- extract local port number ----
uint16_t portNumber=std::stoi(args[1]);

//---- create listen socket ----
//
SOCKET s = crs::socket(PF_INET, SOCK_STREAM, 0);
crs::bind(s, portNumber);
crs::listen(s);
//
// Créer avec ``crs::socket()'' une socket TCP, et utiliser ``crs::bind()''
// pour qu'elle soit associée au port ``portNumber'' de la machine.
// Il s'agit d'une socket d'écoute ; ceci sera spécifié par l'appel à
// ``crs::listen()''.
//

// ...

for(;;)
  {
  uint32_t FromAddr;
  uint16_t FromPort;
  //---- accept and display new connection ----
  crs::write(STDOUT_FILENO,crs::txt(
             "host '%' waiting for a new connection on port '%'...\n",
             crs::gethostname(),portNumber));
  //
  auto dialogSocket = crs::accept(s,FromAddr,FromPort);
  
  //
  // Accepter, à l'aide de ``crs::accept()'', la prochaine connexion sur la
  // socket d'écoute.
  // Cette opération fait apparaître une socket de dialogue TCP.
  // Afficher les coordonnées du client qui est à l'origine de cette
  // connexion.
  //

  // ...

  for(;;) // as long as dialog can go on...
    {

    //---- receive and display request from client ----
    crs::write(STDOUT_FILENO,crs::txt(
               "host '%' waiting for a TCP message from client...\n",
               crs::gethostname()));
    std::string request;
    

    request = crs::recv(dialogSocket,0x100);  
    auto reply=crs::txt("server received % bytes\n",request.size());
    
    
    if(request.empty()){ break;}
    crs::send(dialogSocket,reply);
    //
    // Recevoir dans ``request'' du texte depuis la socket de dialogue TCP
    // avec ``crs::recv()''.
    // Si le texte reçu est vide, cela signifie que la fin-de-fichier est
    // atteinte (le client a fermé la connexion) ; il suffit de quitter la
    // boucle de dialogue avec ``break;'', sinon afficher le texte reçu.

    // Envoyer avec ``crs::send()'', la réponse ``reply'' au client à
    // travers la socket de dialogue TCP qui nous y relie.
    //

    // ...
    }

  //---- close dialog socket ----
  //
    crs::close(dialogSocket);
  //
  // Fermer la socket de dialogue avec ``crs::close()''.
  //

  // ...

    crs::write(STDOUT_FILENO,"client disconnected\n");
  }

//---- close listen socket ----
//
crs::close(s);
//
// Fermer la socket d'écoute avec ``crs::close()''.
// Même si cette portion de code n'est jamais atteinte ici (dans ce programme
// simpliste), il faut toujours se poser la question de la fermeture !
//

// ...

return 0;
}

//----------------------------------------------------------------------------
