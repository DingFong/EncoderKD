# Encoder-KD
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>


<!-- PROJECT SHIELDS -->

<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<!-- <br />
<div align="center">
  <a href="https://github.com/DingFong/DroneDectection_yolo">
    <img src="readme_images/logo.jpg" alt="Logo" width="80" height="80">
  </a>
</div> -->



<!-- TABLE OF CONTENTS -->
<!-- <details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->


<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
* Package
  ```sh
  pip install -r -requirements.txt
  ```

* dataset imagenet-1k  
  Create folder "data"  
  imagenet-1k - https://www.image-net.org/

* teacher weight  
  Create folder "teacher_weights"  
  https://connecthkuhk-my.sharepoint.com/personal/ruifeihe_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fruifeihe%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fproj%2FKDEP%5FCVPR2022%2Fteacher%5Fweights&ga=1

### DDP Training
```sh
   torchrun --nproc_per_node=4 train_autoencoder.py --num_workers 4
```
<!-- ### Predict

```sh
   python3 detect.py --source DroneDataset/yolo_format/images/test/ --weights runs/train/yolov7-e6e_drone4/weights/best.pt --conf 0.1 --name yolov7-e6e_drone --save-txt --save-conf --img-size 1280
``` -->

<!-- ### Generate submission file
--file_model: select model prediciton result you want, whcih save in folder "runs/detect".
--threshold: set confidence threshold to filter out low confidence result.
```sh
  python3 filter_low_probability.py --file_model yolov7-e6e_drone --threshold 0.2
``` -->
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
<!-- ## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ROADMAP
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- LICENSE
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact
Fred - a890702000@gmail.com
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
<p align="right">(<a href="#readme-top">back to top</a>)</p>
