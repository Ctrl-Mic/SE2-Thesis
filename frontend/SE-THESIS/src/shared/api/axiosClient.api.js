import axios from "axios";
// ?So we make an axios client here so that we have one client handling all the request for the server.
// ?Meaning we dont need to always call the url and create the header everytime we request to server. We
// ?just call this then indicate in which url are we going.

// ?The second part, the interceptor handles the server status. So if ever na down yung server while we
// ?are in the login page, The interceptor will instead send a status of server_down if he cant contact
// ?that server url. This prevent page breaking pag biglang nag down yung server. Parang naka avr yung
// ?server

const axiosClient = axios.create({
  baseURL: "http://localhost:4000",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 5000,
});

axiosClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (!error.response) {
      return Promise.reject({
        type: "server_down",
        message: "Server is unreachable",
      });
    }

    return Promise.reject(error);
  },
);

export default axiosClient;
